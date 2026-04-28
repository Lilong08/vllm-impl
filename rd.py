from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from hashlib import sha256
from typing import Optional, Hashable


BlockHash = bytes


def hash_block(
    parent_hash: Optional[BlockHash],
    tokens: tuple[int, ...],
    extra_keys: tuple[Hashable, ...] = (),
) -> BlockHash:
    """
    简化版 vLLM block hash:
    hash(parent_hash, block_tokens, extra_keys)

    extra_keys 可用于区分 LoRA、tenant salt、多模态输入 hash 等。
    """
    h = sha256()
    h.update(parent_hash or b"<ROOT>")
    h.update(repr(tokens).encode())
    h.update(repr(extra_keys).encode())
    return h.digest()


@dataclass
class KVBlock:
    block_id: int
    ref_cnt: int = 0
    block_hash: Optional[BlockHash] = None
    tokens: tuple[int, ...] = ()

    @property
    def is_cached(self) -> bool:
        return self.block_hash is not None


class SimpleKVBlockManager:
    """
    简化版 KV cache block manager.

    它不管理真实 GPU tensor，只管理“物理 block id”。
    实际系统中，block_id 会对应到预分配 KV cache tensor 的某段显存。
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.block_size = block_size
        self.blocks = [KVBlock(i) for i in range(num_blocks)]

        # 空闲 block 队列：OrderedDict 用来模拟 LRU free queue。
        # key 是 block_id，value 是 KVBlock。
        self.free_blocks: OrderedDict[int, KVBlock] = OrderedDict(
            (b.block_id, b) for b in self.blocks
        )

        # prefix cache: block_hash -> KVBlock
        # 简化处理：相同 hash 只保留一个 block。
        # vLLM 里允许 duplicated blocks，因为 block table append-only。
        self.cache: dict[BlockHash, KVBlock] = {}

        # request_id -> block list
        self.request_blocks: dict[str, list[KVBlock]] = {}

        # request_id -> token list
        self.request_tokens: dict[str, list[int]] = {}

    def _pop_free_block(self) -> KVBlock:
        """
        从 LRU free queue 头部拿一个 block。
        如果它之前是 cached block，需要先从 cache 中淘汰。
        """
        if not self.free_blocks:
            raise RuntimeError("No free KV cache blocks")

        _, block = self.free_blocks.popitem(last=False)

        if block.block_hash is not None:
            self.cache.pop(block.block_hash, None)
            block.block_hash = None
            block.tokens = ()

        block.ref_cnt = 1
        return block

    def _touch_cached_block(self, block: KVBlock) -> None:
        """
        cache hit 后，增加 ref_cnt。
        如果 block 在 free queue 中，说明它当前没人用但可被复用；
        命中后要从 free queue 移除，避免被淘汰。
        """
        if block.block_id in self.free_blocks:
            del self.free_blocks[block.block_id]
        block.ref_cnt += 1

    def _free_block(self, block: KVBlock) -> None:
        """
        释放一个 block 的引用。
        ref_cnt 归零后放回 free queue 尾部，作为较新的 LRU 候选。
        """
        if block.ref_cnt <= 0:
            raise RuntimeError(f"Block {block.block_id} ref_cnt underflow")

        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self.free_blocks[block.block_id] = block

    def _split_full_blocks(self, tokens: list[int]) -> list[tuple[int, ...]]:
        """
        只返回 full blocks。
        prefix caching 只复用满 block。
        """
        n_full = len(tokens) // self.block_size
        return [
            tuple(tokens[i * self.block_size : (i + 1) * self.block_size])
            for i in range(n_full)
        ]

    def lookup_prefix(
        self,
        tokens: list[int],
        extra_keys: tuple[Hashable, ...] = (),
    ) -> tuple[list[KVBlock], int, Optional[BlockHash]]:
        """
        查找最长连续 prefix cache hit。

        返回:
        - 命中的 blocks
        - 命中的 token 数
        - 最后一个命中 block 的 hash，供后续新 block 继续链式 hash
        """
        hits: list[KVBlock] = []
        parent_hash: Optional[BlockHash] = None

        for block_tokens in self._split_full_blocks(tokens):
            bh = hash_block(parent_hash, block_tokens, extra_keys)
            block = self.cache.get(bh)
            if block is None:
                break

            hits.append(block)
            parent_hash = bh

        return hits, len(hits) * self.block_size, parent_hash

    def allocate_request(
        self,
        request_id: str,
        tokens: list[int],
        extra_keys: tuple[Hashable, ...] = (),
    ) -> tuple[list[int], int]:
        """
        为新请求分配 KV blocks，并复用已有 prefix cache。

        返回:
        - block table，即物理 block ids
        - computed_tokens，即已复用、无需重新 prefill 的 token 数
        """
        if request_id in self.request_blocks:
            raise RuntimeError(f"request {request_id!r} already exists")

        hit_blocks, computed_tokens, parent_hash = self.lookup_prefix(
            tokens, extra_keys
        )

        for block in hit_blocks:
            self._touch_cached_block(block)

        allocated: list[KVBlock] = list(hit_blocks)

        full_blocks = self._split_full_blocks(tokens)
        start_block_idx = len(hit_blocks)

        # 只给 full blocks 分配并缓存。
        # 非满尾块在真实 vLLM 中也会占 slot，但不能进入 prefix cache；
        # 这里为了突出复用逻辑，先不分配 partial block。
        for block_tokens in full_blocks[start_block_idx:]:
            block = self._pop_free_block()
            block.tokens = block_tokens

            bh = hash_block(parent_hash, block_tokens, extra_keys)
            block.block_hash = bh
            self.cache[bh] = block

            allocated.append(block)
            parent_hash = bh

        self.request_blocks[request_id] = allocated
        self.request_tokens[request_id] = list(tokens)

        return [b.block_id for b in allocated], computed_tokens

    def append_tokens(
        self,
        request_id: str,
        new_tokens: list[int],
        extra_keys: tuple[Hashable, ...] = (),
    ) -> tuple[list[int], list[int]]:
        """
        给 running request 追加 token。

        简化策略:
        - 如果追加后产生新的 full block，就分配新 block 并缓存。
        - 不模拟真实 paged-attention 的 slot 级写入。
        """
        if request_id not in self.request_blocks:
            raise RuntimeError(f"unknown request {request_id!r}")

        old_tokens = self.request_tokens[request_id]
        old_full = len(old_tokens) // self.block_size

        old_tokens.extend(new_tokens)
        new_full_blocks = self._split_full_blocks(old_tokens)

        allocated_now: list[KVBlock] = []

        # 找到当前 request 最后一个已缓存 full block 的 hash
        parent_hash: Optional[BlockHash] = None
        for block_tokens in new_full_blocks[:old_full]:
            parent_hash = hash_block(parent_hash, block_tokens, extra_keys)

        for block_tokens in new_full_blocks[old_full:]:
            block = self._pop_free_block()
            block.tokens = block_tokens

            bh = hash_block(parent_hash, block_tokens, extra_keys)
            block.block_hash = bh
            self.cache[bh] = block

            self.request_blocks[request_id].append(block)
            allocated_now.append(block)
            parent_hash = bh

        return (
            [b.block_id for b in self.request_blocks[request_id]],
            [b.block_id for b in allocated_now],
        )

    def free_request(self, request_id: str) -> None:
        """
        释放请求占用的 blocks。

        反向释放：尾部 block 通常包含更长 prefix，复用概率更低，
        因此更早进入 LRU 淘汰候选。
        """
        blocks = self.request_blocks.pop(request_id, None)
        self.request_tokens.pop(request_id, None)

        if blocks is None:
            return

        for block in reversed(blocks):
            self._free_block(block)

    def stats(self) -> dict[str, int]:
        return {
            "num_blocks": len(self.blocks),
            "num_free_blocks": len(self.free_blocks),
            "num_cached_blocks": len(self.cache),
            "num_active_requests": len(self.request_blocks),
        }
    
if __name__ == "__main__":
    mgr = SimpleKVBlockManager(num_blocks=4, block_size=4)

    # 请求 1: tokens = [1..8]，两个 full blocks
    b1, hit1 = mgr.allocate_request("req1", [1, 2, 3, 4, 5, 6, 7, 8])
    print("req1 blocks:", b1, "computed_tokens:", hit1)
    print("stats:", mgr.stats())

    mgr.free_request("req1")
    print("after free req1:", mgr.stats())

    # 请求 2: 前 8 个 token 完全相同，可以复用 req1 的两个 cached blocks
    b2, hit2 = mgr.allocate_request("req2", [1, 2, 3, 4, 5, 6, 7, 8, 99])
    print("req2 blocks:", b2, "computed_tokens:", hit2)
    print("stats:", mgr.stats())

    # 输出大致类似:
    # req1 blocks: [0, 1] computed_tokens: 0
    # after free req1: cached blocks 仍然在，但 ref_cnt=0，等待复用或淘汰
    # req2 blocks: [0, 1] computed_tokens: 8
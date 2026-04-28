from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import itertools


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


_request_id_gen = itertools.count()


@dataclass
class Request:
    prompt_tokens: list[int]
    max_new_tokens: int

    request_id: int = field(default_factory=lambda: next(_request_id_gen))
    status: RequestStatus = RequestStatus.WAITING

    generated_tokens: list[int] = field(default_factory=list)
    num_computed_prompt_tokens: int = 0
    kv_blocks: int = 0

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def is_prefill_done(self) -> bool:
        return self.num_computed_prompt_tokens >= len(self.prompt_tokens)

    @property
    def is_finished(self) -> bool:
        return len(self.generated_tokens) >= self.max_new_tokens

    def next_token_count(self) -> int:
        """
        当前调度轮需要计算多少 token。
        - prefill 阶段：计算剩余 prompt token
        - decode 阶段：每轮生成 1 个 token
        """
        if not self.is_prefill_done:
            return len(self.prompt_tokens) - self.num_computed_prompt_tokens
        return 1


class KVCacheManager:
    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.used_blocks = 0

    def required_blocks(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        blocks = self.required_blocks(num_tokens)
        return self.used_blocks + blocks <= self.num_blocks

    def allocate_for_request(self, req: Request, new_total_tokens: int) -> bool:
        """
        根据请求新的 token 总量补充分配 KV blocks。
        """
        old_blocks = req.kv_blocks
        new_blocks = self.required_blocks(new_total_tokens)
        extra_blocks = new_blocks - old_blocks

        if extra_blocks <= 0:
            return True

        if self.used_blocks + extra_blocks > self.num_blocks:
            return False

        self.used_blocks += extra_blocks
        req.kv_blocks = new_blocks
        return True

    def free(self, req: Request):
        self.used_blocks -= req.kv_blocks
        req.kv_blocks = 0


@dataclass
class SchedulerOutput:
    scheduled_requests: list[Request]
    num_batched_tokens: int


class Scheduler:
    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        block_size: int,
        num_kv_blocks: int,
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        self.kv_cache = KVCacheManager(
            block_size=block_size,
            num_blocks=num_kv_blocks,
        )

    def add_request(self, req: list[Request]):
        self.waiting.extend(req)

    def schedule(self) -> SchedulerOutput:
        scheduled = []
        token_budget = self.max_num_batched_tokens
        seq_budget = self.max_num_seqs

        # 1. 先调度 running 请求
        still_running = []

        for req in self.running:
            if seq_budget <= 0:
                still_running.append(req)
                continue

            need_tokens = req.next_token_count()

            if need_tokens > token_budget:
                still_running.append(req)
                continue

            new_total_tokens = req.total_tokens + need_tokens

            if not self.kv_cache.allocate_for_request(req, new_total_tokens):
                still_running.append(req)
                continue

            scheduled.append(req)
            token_budget -= need_tokens
            seq_budget -= 1
            still_running.append(req)

        self.running = still_running

        # 2. 再从 waiting 队列中加入新请求
        while self.waiting and seq_budget > 0:
            req = self.waiting[0]
            need_tokens = req.next_token_count()

            if need_tokens > token_budget:
                break

            if not self.kv_cache.allocate_for_request(req, len(req.prompt_tokens)):
                break

            self.waiting.popleft()
            req.status = RequestStatus.RUNNING

            scheduled.append(req)
            self.running.append(req)

            token_budget -= need_tokens
            seq_budget -= 1

        return SchedulerOutput(
            scheduled_requests=scheduled,
            num_batched_tokens=self.max_num_batched_tokens - token_budget,
        )


class core_engine:

    def __init__(self):
        self.scheduler = Scheduler(
            max_num_seqs=3,
            max_num_batched_tokens=16,
            block_size=4,
            num_kv_blocks=32,
        )

    def has_unfinished_requests(self) -> bool:
        return bool(self.scheduler.waiting or self.scheduler.running)

    def step(self):
        """
        模拟一次模型 forward。
        """
        output = self.scheduler.schedule()

        for req in output.scheduled_requests:
            if not req.is_prefill_done:
                # prefill 阶段：一次性处理剩余 prompt
                req.num_computed_prompt_tokens = len(req.prompt_tokens)
            else:
                # decode 阶段：模拟生成一个 token
                fake_token = 1000 + len(req.generated_tokens)
                req.generated_tokens.append(fake_token)

            if req.is_finished:
                req.status = RequestStatus.FINISHED
                self.scheduler.kv_cache.free(req)

        self.scheduler.running = [
            req for req in self.scheduler.running
            if req.status != RequestStatus.FINISHED
        ]

        return output

class LLM:
    
    def __init__(self):
        self.core_engine = core_engine()

    def generate(self, request : list[Request]):
        self.core_engine.scheduler.add_request(request)
        self.run_engine()

    def run_engine(self):
        step_id = 0
        while self.core_engine.has_unfinished_requests():
            out = self.core_engine.step()
            print(f"\nStep {step_id}")
            print("scheduled:", [r.request_id for r in out.scheduled_requests])
            print("batched tokens:", out.num_batched_tokens)
            print("kv used blocks:", self.core_engine.scheduler.kv_cache.used_blocks)
            step_id += 1


if __name__ == "__main__": 
    llm = LLM()

    print('init the scheduler')
    r1 = Request(prompt_tokens=[1, 2, 3, 4], max_new_tokens=3)
    r2 = Request(prompt_tokens=[5, 6, 7], max_new_tokens=2)
    r3 = Request(prompt_tokens=[8, 9, 10, 11, 12], max_new_tokens=4)
    llm.generate([r1,r2,r3])
    r4 = Request(prompt_tokens=[1, 2, 3, 4], max_new_tokens=3)
    r5 = Request(prompt_tokens=[5, 6, 7], max_new_tokens=2)
    r6 = Request(prompt_tokens=[8, 9, 10, 11, 12], max_new_tokens=4)
    llm.generate([r4,r5,r6])

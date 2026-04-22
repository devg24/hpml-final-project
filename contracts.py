from dataclasses import dataclass
from typing import Literal


@dataclass
class AgentSpec:
    id: int
    persona: str  # the task prompt / system instruction


@dataclass
class ExperimentConfig:
    agents: list[AgentSpec]
    shared_code_prefix: str
    max_new_tokens: int = 256
    model_path: str = "./qwen-7b"
    device: str = "cuda:0"


@dataclass
class AgentResult:
    agent_id: int
    status: Literal["success", "oom", "error"]
    ttft_seconds: float | None = None
    total_time_seconds: float | None = None
    throughput_tps: float | None = None
    output_tokens: int | None = None
    perplexity: float | None = None
    error_trace: str | None = None


@dataclass
class AggregateMetrics:
    success_rate: float
    oom_count: int
    avg_ttft: float
    p50_ttft: float
    p99_ttft: float
    avg_throughput: float
    avg_perplexity: float
    base_code_perplexity: float
    peak_vram_gb: float


@dataclass
class ExperimentResult:
    backend: str  # "hf_bf16" | "hf_nf4" | "vllm" | "vllm_awq" | "speculative"
    config: ExperimentConfig
    agent_results: list[AgentResult]
    agg: AggregateMetrics
    wall_time_seconds: float
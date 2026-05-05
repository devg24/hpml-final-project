"""
backends/hf_vllm.py
--------------------
vLLM inference backend with prefix KV-cache enabled.
"""

import asyncio
import math
import time
import traceback

import torch
import torch.cuda
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.outputs import RequestOutput

from contracts import AgentResult, ExperimentConfig, ExperimentResult
from backends.utils import compute_agg

BACKEND_NAME = "vllm"


def _engine_args(cfg: ExperimentConfig) -> AsyncEngineArgs:
    return AsyncEngineArgs(
        model=cfg.model_path,
        dtype="bfloat16",
        enable_prefix_caching=True,   # RadixAttention — prefix KV shared across agents
        max_model_len=8192,           # Increased to accommodate the 6k token prefix
        gpu_memory_utilization=0.80,  # Lowered to 80% to leave room for prefill activations
        enforce_eager=True,           # Disables CUDA graphs to save ~1GB of VRAM
        trust_remote_code=True,
        disable_log_stats=True,       # Correct argument to disable periodic stat logging
    )


def _ppl_from_logprobs(logprobs_list) -> float | None:
    """Perplexity from a list of per-token logprob dicts returned by vLLM."""
    if not logprobs_list:
        return None
    total_nll = 0.0
    count = 0
    for lp_dict in logprobs_list:
        if lp_dict is None:
            continue
        # Take the chosen token's logprob
        lp = next(iter(lp_dict.values())).logprob
        total_nll -= lp
        count += 1
    return math.exp(total_nll / count) if count > 0 else None


async def _base_ppl(engine: AsyncLLMEngine, cfg: ExperimentConfig) -> float | None:
    """Score the shared prefix via prompt_logprobs to get base-code perplexity."""
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
    )
    final: RequestOutput | None = None
    async for out in engine.generate(cfg.shared_code_prefix, params, request_id="base-ppl"):
        final = out

    if final is None:
        return None
    return _ppl_from_logprobs(final.prompt_logprobs)


async def _generate_one(
    agent,
    engine: AsyncLLMEngine,
    cfg: ExperimentConfig,
) -> AgentResult:
    prompt = (
        f"Code:\n{cfg.shared_code_prefix}\n\n"
        f"System: {agent.persona}\n\n"
        f"Review:"
    )
    params = SamplingParams(
        max_tokens=cfg.max_new_tokens,
        temperature=0.0,
        logprobs=1,
    )
    start_time = time.time()
    ttft: float | None = None
    final: RequestOutput | None = None

    try:
        async for output in engine.generate(prompt, params, request_id=f"agent-{agent.id}"):
            if ttft is None and output.outputs and output.outputs[0].token_ids:
                ttft = time.time() - start_time
            final = output

        end_time = time.time()
        total_time = end_time - start_time

        if final is None or not final.outputs:
            raise RuntimeError("No output from engine")

        completion = final.outputs[0]
        output_tokens = len(completion.token_ids)
        throughput = output_tokens / total_time if total_time > 0 else 0.0
        perplexity = _ppl_from_logprobs(completion.logprobs)

        return AgentResult(
            agent_id=agent.id,
            status="success",
            ttft_seconds=ttft if ttft is not None else total_time,
            total_time_seconds=total_time,
            throughput_tps=throughput,
            output_tokens=output_tokens,
            perplexity=perplexity,
        )

    except Exception:
        return AgentResult(
            agent_id=agent.id,
            status="error",
            error_trace=traceback.format_exc(),
        )


async def run_vllm(cfg: ExperimentConfig) -> ExperimentResult:
    torch.cuda.reset_peak_memory_stats(cfg.device)

    print(f"[{BACKEND_NAME}] Starting AsyncLLMEngine (prefix_caching=True) ...")
    engine = AsyncLLMEngine.from_engine_args(_engine_args(cfg))

    # Compute base-code perplexity first
    print(f"[{BACKEND_NAME}] Calculating base code perplexity ...")
    base_ppl = await _base_ppl(engine, cfg)
    print(
        f"[{BACKEND_NAME}] Base PPL: {base_ppl:.4f}"
        if base_ppl is not None
        else f"[{BACKEND_NAME}] Base PPL: N/A"
    )

    # Dispatch all agents concurrently
    wall_start = time.time()
    tasks = [_generate_one(agent, engine, cfg) for agent in cfg.agents]
    agent_results = list(await asyncio.gather(*tasks))
    wall_time = time.time() - wall_start

    peak_vram = torch.cuda.max_memory_allocated(cfg.device) / 1e9

    # Shut down/free VRAM
    del engine
    torch.cuda.empty_cache()

    agg = compute_agg(agent_results, base_ppl or 0.0, peak_vram, len(cfg.agents))
    return ExperimentResult(
        backend=BACKEND_NAME,
        config=cfg,
        agent_results=agent_results,
        agg=agg,
        wall_time_seconds=wall_time,
    )
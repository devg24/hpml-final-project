"""
backends/hf_quantized.py
------------------------
HuggingFace Transformers inference with INT4 NF4 quantization via bitsandbytes.
Generation harness is identical to hf_baseline; only the model-load path differs.
"""

import asyncio
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from contracts import AgentResult, ExperimentConfig, ExperimentResult
from backends.utils import (
    TTFTStreamer,
    compute_agg,
    compute_generation_perplexities,
    calculate_perplexity,
)

BACKEND_NAME = "hf_nf4"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load(cfg: ExperimentConfig):
    print(f"[{BACKEND_NAME}] Loading tokenizer + model ({cfg.model_path}) with NF4 ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # quantize the quantization constants too
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        quantization_config=bnb_config,
        device_map="auto",               # bitsandbytes requires device_map="auto"
        trust_remote_code=True,
    )
    model.eval()
    print(f"[{BACKEND_NAME}] Model loaded.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Single-agent generation  (synchronous — called inside a thread)
# ---------------------------------------------------------------------------

def _generate_one(
    agent: "AgentSpec",  # noqa: F821
    model,
    tokenizer,
    cfg: ExperimentConfig,
) -> tuple[AgentResult, torch.Tensor | None, int | None]:
    full_prompt = (
        f"Code:\n{cfg.shared_code_prefix}\n\n"
        f"System: {agent.persona}\n\n"
        f"Review:"
    )
    # NF4 models loaded with device_map="auto" may span devices; use the
    # device of the first parameter as the input target.
    device = next(model.parameters()).device
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    streamer = TTFTStreamer()
    start_time = time.time()

    cuda_stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(cuda_stream):
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                streamer=streamer,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        end_time = time.time()

        total_time = end_time - start_time
        ttft = (
            streamer.first_token_time - start_time
            if streamer.first_token_time
            else total_time
        )
        input_len = inputs["input_ids"].shape[-1]
        output_tokens = len(outputs[0]) - input_len
        throughput = output_tokens / total_time if total_time > 0 else 0.0

        full_ids_cpu = outputs[0].unsqueeze(0).cpu()

        result = AgentResult(
            agent_id=agent.id,
            status="success",
            ttft_seconds=ttft,
            total_time_seconds=total_time,
            throughput_tps=throughput,
            output_tokens=output_tokens,
        )
        return result, full_ids_cpu, input_len

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return (
            AgentResult(
                agent_id=agent.id,
                status="oom",
                error_trace=traceback.format_exc(),
            ),
            None,
            None,
        )
    except Exception:
        return (
            AgentResult(
                agent_id=agent.id,
                status="error",
                error_trace=traceback.format_exc(),
            ),
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_hf_quantized(cfg: ExperimentConfig) -> ExperimentResult:
    torch.cuda.reset_peak_memory_stats(cfg.device)
    model, tokenizer = _load(cfg)

    # Base-code perplexity
    print(f"[{BACKEND_NAME}] Calculating base code perplexity ...")
    device = next(model.parameters()).device
    base_inputs = tokenizer(
        cfg.shared_code_prefix, return_tensors="pt"
    ).to(device)
    base_ppl = calculate_perplexity(model, base_inputs["input_ids"])
    print(f"[{BACKEND_NAME}] Base PPL: {base_ppl:.4f}")

    # Dispatch all agents concurrently
    wall_start = time.time()
    tasks = [
        asyncio.to_thread(_generate_one, agent, model, tokenizer, cfg)
        for agent in cfg.agents
    ]
    raw = await asyncio.gather(*tasks)

    # Sequential perplexity pass
    print(f"[{BACKEND_NAME}] Computing per-agent perplexity ...")
    pending = [(res, ids, ilen) for res, ids, ilen in raw]
    agent_results = [res for res, _, _ in raw]
    compute_generation_perplexities(model, agent_results, pending, str(device))

    wall_time = time.time() - wall_start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9  # "auto" may span devices

    del model
    torch.cuda.empty_cache()

    agg = compute_agg(agent_results, base_ppl, peak_vram, len(cfg.agents))
    return ExperimentResult(
        backend=BACKEND_NAME,
        config=cfg,
        agent_results=agent_results,
        agg=agg,
        wall_time_seconds=wall_time,
    )
"""
backends/hf_speculative.py
--------------------------
HuggingFace Transformers inference using Speculative Decoding.
Requires a smaller "draft" model to generate token proposals for the target model.
"""

import asyncio
import concurrent.futures
import time
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from contracts import AgentResult, ExperimentConfig, ExperimentResult
from backends.utils import (
    TTFTStreamer,
    compute_agg,
    compute_generation_perplexities,
    calculate_perplexity,
)

BACKEND_NAME = "hf_speculative"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load(cfg: ExperimentConfig):
    print(f"[{BACKEND_NAME}] Loading target tokenizer + model ({cfg.model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

    # Load the main/target model
    target_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    target_model.eval()
    
    # Load the draft model and its tokenizer (Universal Assisted Decoding)
    print(f"[{BACKEND_NAME}] Loading draft tokenizer + model ({cfg.draft_model_path})...")
    draft_tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model_path, trust_remote_code=True)
    draft_model = AutoModelForCausalLM.from_pretrained(
        cfg.draft_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    draft_model.eval()

    print(f"[{BACKEND_NAME}] Models loaded.")
    return target_model, draft_model, tokenizer, draft_tokenizer


# ---------------------------------------------------------------------------
# Single-agent generation (synchronous — called inside a thread)
# ---------------------------------------------------------------------------

def _generate_one(
    agent,
    target_model,
    draft_model,
    tokenizer,
    draft_tokenizer,
    cfg: ExperimentConfig,
) -> tuple[AgentResult, torch.Tensor | None, int | None]:
    full_prompt = (
        f"Code:\n{cfg.shared_code_prefix}\n\n"
        f"System: {agent.persona}\n\n"
        f"Review:"
    )
    
    device = next(target_model.parameters()).device
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    streamer = TTFTStreamer()
    start_time = time.time()

    cuda_stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(cuda_stream):
            outputs = target_model.generate(
                **inputs,
                assistant_model=draft_model, # <-- Enables Speculative Decoding
                tokenizer=tokenizer,         # Standard re-tokenization logic
                assistant_tokenizer=draft_tokenizer, # <-- Enables Universal Assisted Decoding
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

async def run_speculative(cfg: ExperimentConfig) -> ExperimentResult:
    torch.cuda.reset_peak_memory_stats(cfg.device)
    target_model, draft_model, tokenizer, draft_tokenizer = _load(cfg)

    # Base-code perplexity
    print(f"[{BACKEND_NAME}] Calculating base code perplexity ...")
    device = next(target_model.parameters()).device
    base_inputs = tokenizer(
        cfg.shared_code_prefix, return_tensors="pt"
    ).to(device)
    base_ppl = calculate_perplexity(target_model, base_inputs["input_ids"])
    print(f"[{BACKEND_NAME}] Base PPL: {base_ppl:.4f}")

    # Dispatch all agents concurrently with a custom executor
    wall_start = time.time()
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cfg.agents)) as executor:
        tasks = [
            loop.run_in_executor(executor, _generate_one, agent, target_model, draft_model, tokenizer, draft_tokenizer, cfg)
            for agent in cfg.agents
        ]
        raw = await asyncio.gather(*tasks)

    # Sequential perplexity pass
    print(f"[{BACKEND_NAME}] Computing per-agent perplexity ...")
    pending = list(raw)
    agent_results = [res for res, _, _ in raw]
    compute_generation_perplexities(pending, target_model, str(device))

    wall_time = time.time() - wall_start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    # Free memory
    del target_model
    del draft_model
    torch.cuda.empty_cache()

    agg = compute_agg(agent_results, base_ppl, peak_vram, len(cfg.agents))
    return ExperimentResult(
        backend=BACKEND_NAME,
        config=cfg,
        agent_results=agent_results,
        agg=agg,
        wall_time_seconds=wall_time,
    )
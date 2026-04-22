"""
run_all.py
----------
Orchestrator.  Loads shared setup, then runs each registered backend
sequentially (one GPU — we free VRAM between experiments), logging
every result to a shared WandB run.

Usage:
    python run_all.py                        # all backends, 50 agents
    python run_all.py --backends hf_bf16 hf_nf4
    python run_all.py --n-agents 15
"""

import argparse
import asyncio
import inspect
import json
import os

import torch
import wandb
import torch.nn.modules.module as module

from contracts import AgentSpec, ExperimentConfig
from backends.hf_baseline  import run_hf_baseline
from backends.hf_quantized import run_hf_quantized
from backends.hf_speculative import run_speculative

# ---------------------------------------------------------------------------
# Backend registry  — add new backends here, nothing else needs to change
# ---------------------------------------------------------------------------

REGISTRY: dict[str, callable] = {
    "hf_bf16":     run_hf_baseline,
    "hf_nf4":      run_hf_quantized,
    # "vllm":        run_vllm,       # coming soon
    # "vllm_awq":    run_vllm_awq,   # coming soon
    "speculative": run_speculative, # coming soon
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH      = "./qwen-7b"
AGENTS_FILE     = "agents_config.json"
WANDB_PROJECT   = "hpml-final-project"
WANDB_ENTITY    = "ak5446-columbia-university"
MAX_NEW_TOKENS  = 256
DEVICE          = "cuda:0"

# ---------------------------------------------------------------------------
# Shared code prefix  (same as original — first 600 lines of torch.nn.Module)
# ---------------------------------------------------------------------------

def _load_code_prefix() -> str:
    src = inspect.getsource(module)
    return "\n".join(src.splitlines()[:600])


# ---------------------------------------------------------------------------
# WandB logging
# ---------------------------------------------------------------------------

def _log_result(result) -> None:
    b = result.backend
    agg = result.agg

    # Per-agent metrics
    for r in result.agent_results:
        if r.status == "success":
            wandb.log({
                f"{b}/agent_{r.agent_id}/ttft":        r.ttft_seconds,
                f"{b}/agent_{r.agent_id}/throughput":  r.throughput_tps,
                f"{b}/agent_{r.agent_id}/perplexity":  r.perplexity,
                f"{b}/agent_{r.agent_id}/output_tokens": r.output_tokens,
            })

    # Aggregate metrics
    wandb.log({
        f"{b}/agg/success_rate":          agg.success_rate,
        f"{b}/agg/oom_count":             agg.oom_count,
        f"{b}/agg/avg_ttft":              agg.avg_ttft,
        f"{b}/agg/p50_ttft":              agg.p50_ttft,
        f"{b}/agg/p99_ttft":              agg.p99_ttft,
        f"{b}/agg/avg_throughput":        agg.avg_throughput,
        f"{b}/agg/avg_perplexity":        agg.avg_perplexity,
        f"{b}/agg/base_code_perplexity":  agg.base_code_perplexity,
        f"{b}/agg/peak_vram_gb":          agg.peak_vram_gb,
        f"{b}/agg/wall_time_seconds":     result.wall_time_seconds,
    })


def _print_summary(result) -> None:
    agg = result.agg
    n = len(result.agent_results)
    print(f"\n{'='*55}")
    print(f"  Backend : {result.backend}")
    print(f"  Agents  : {int(agg.success_rate * n)}/{n} succeeded  |  {agg.oom_count} OOM")
    print(f"  TTFT    : avg {agg.avg_ttft:.2f}s  p50 {agg.p50_ttft:.2f}s  p99 {agg.p99_ttft:.2f}s")
    print(f"  Thruput : {agg.avg_throughput:.1f} tok/s")
    print(f"  PPL     : agent avg {agg.avg_perplexity:.3f}  |  base code {agg.base_code_perplexity:.3f}")
    print(f"  VRAM    : {agg.peak_vram_gb:.2f} GB peak")
    print(f"  Wall    : {result.wall_time_seconds:.1f}s")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(backends_to_run: list[str], n_agents: int) -> None:
    # -- Load shared setup --------------------------------------------------
    if not os.path.exists(AGENTS_FILE):
        raise FileNotFoundError(f"{AGENTS_FILE} not found.")

    with open(AGENTS_FILE) as f:
        raw_agents = json.load(f).get("agents", [])

    agents = [
        AgentSpec(id=a["id"], persona=a["persona"])
        for a in raw_agents[:n_agents]
    ]
    if len(agents) < n_agents:
        print(f"Warning: only {len(agents)} agents available (requested {n_agents}).")

    shared_code_prefix = _load_code_prefix()
    print(f"Loaded {len(agents)} agents | prefix length: {len(shared_code_prefix)} chars")

    cfg = ExperimentConfig(
        agents=agents,
        shared_code_prefix=shared_code_prefix,
        max_new_tokens=MAX_NEW_TOKENS,
        model_path=MODEL_PATH,
        device=DEVICE,
    )

    # -- WandB init ---------------------------------------------------------
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"comparison_{'_'.join(backends_to_run)}_{n_agents}agents",
        config={
            "model":         "Qwen-7B-Instruct",
            "n_agents":      n_agents,
            "max_new_tokens": MAX_NEW_TOKENS,
            "backends":      backends_to_run,
            "prefix_chars":  len(shared_code_prefix),
        },
    )

    # -- Run each backend sequentially (one GPU) ----------------------------
    all_results = []
    for name in backends_to_run:
        if name not in REGISTRY:
            print(f"[run_all] Unknown backend '{name}', skipping.")
            continue
        print(f"\n>>> Starting backend: {name} ({n_agents} agents) <<<\n")
        result = await REGISTRY[name](cfg)
        _print_summary(result)
        _log_result(result)
        all_results.append(result)

    wandb.finish()

    # -- Final cross-backend comparison table -------------------------------
    print("\n" + "="*55)
    print(f"  {'Backend':<14} {'tok/s':>8} {'p50 TTFT':>10} {'PPL':>8} {'VRAM GB':>9}")
    print("  " + "-"*51)
    for r in all_results:
        a = r.agg
        print(
            f"  {r.backend:<14} {a.avg_throughput:>8.1f}"
            f" {a.p50_ttft:>10.2f} {a.avg_perplexity:>8.3f}"
            f" {a.peak_vram_gb:>9.2f}"
        )
    print("="*55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(REGISTRY.keys()),
        choices=list(REGISTRY.keys()),
        help="Which backends to run (default: all registered)",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=50,
        help="Number of agents to simulate (default: 50)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.backends, args.n_agents))
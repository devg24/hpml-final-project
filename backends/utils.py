import time
import traceback

import numpy as np
import torch
from transformers.generation.streamers import BaseStreamer

from contracts import AgentResult, AggregateMetrics


# ---------------------------------------------------------------------------
# Streamer
# ---------------------------------------------------------------------------

class TTFTStreamer(BaseStreamer):
    """Captures the exact timestamp of the first generated token."""

    def __init__(self):
        self.start_time = time.time()
        self.first_token_time: float | None = None
        self.token_count = 0

    def put(self, value):
        if self.first_token_time is None:
            self.first_token_time = time.time()
        self.token_count += 1

    def end(self):
        pass


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def calculate_perplexity(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    max_chunk_size: int = 2048,
) -> float:
    """OOM-safe perplexity. Chunks long sequences to prevent VRAM overflow."""
    seq_len = input_ids.size(1)

    with torch.no_grad():
        if seq_len <= max_chunk_size:
            lbl = labels if labels is not None else input_ids
            outputs = model(input_ids, labels=lbl)
            return torch.exp(outputs.loss).item()

        nlls: list[float] = []
        valid_tokens = 0
        for i in range(0, seq_len, max_chunk_size):
            chunk_in = input_ids[:, i : i + max_chunk_size]
            chunk_lbl = (
                labels[:, i : i + max_chunk_size]
                if labels is not None
                else chunk_in
            )
            outputs = model(chunk_in, labels=chunk_lbl)
            active = (chunk_lbl != -100).sum().item()
            if active > 0:
                nlls.append(outputs.loss.item() * active)
                valid_tokens += active

    if valid_tokens == 0:
        return 0.0
    return torch.exp(torch.tensor(sum(nlls) / valid_tokens)).item()


# ---------------------------------------------------------------------------
# Aggregate helper  (same logic for every backend)
# ---------------------------------------------------------------------------

def compute_agg(
    results: list[AgentResult],
    base_ppl: float,
    peak_vram_gb: float,
    n_agents: int,
) -> AggregateMetrics:
    successful = [r for r in results if r.status == "success"]
    ttfts = [r.ttft_seconds for r in successful if r.ttft_seconds is not None]
    throughputs = [r.throughput_tps for r in successful if r.throughput_tps is not None]
    perplexities = [r.perplexity for r in successful if r.perplexity is not None]

    return AggregateMetrics(
        success_rate=len(successful) / max(n_agents, 1),
        oom_count=sum(1 for r in results if r.status == "oom"),
        avg_ttft=float(np.mean(ttfts)) if ttfts else 0.0,
        p50_ttft=float(np.percentile(ttfts, 50)) if ttfts else 0.0,
        p99_ttft=float(np.percentile(ttfts, 99)) if ttfts else 0.0,
        avg_throughput=float(np.mean(throughputs)) if throughputs else 0.0,
        avg_perplexity=float(np.mean(perplexities)) if perplexities else 0.0,
        base_code_perplexity=base_ppl,
        peak_vram_gb=peak_vram_gb,
    )


# ---------------------------------------------------------------------------
# Shared perplexity pass  (runs after generation, on CPU-offloaded tensors)
# ---------------------------------------------------------------------------

def compute_generation_perplexities(
    model,
    results: list[AgentResult],
    pending: list[tuple[AgentResult, torch.Tensor, int]],  # (result, full_ids_cpu, input_len)
    device: str,
) -> None:
    """
    Mutates result.perplexity in-place.  Accepts the list of
    (AgentResult, full_ids_cpu, input_len) triples produced during generation.
    Runs sequentially to avoid stacking large tensors on GPU simultaneously.
    """
    for result, full_ids_cpu, input_len in pending:
        if result.status != "success":
            continue
        try:
            full_ids = full_ids_cpu.to(device)
            labels = full_ids.clone()
            labels[:, :input_len] = -100
            result.perplexity = calculate_perplexity(model, full_ids, labels=labels)
        except Exception:
            result.perplexity = None
            result.error_trace = (result.error_trace or "") + "\nPPL: " + traceback.format_exc()
        finally:
            del full_ids, labels
            torch.cuda.empty_cache()
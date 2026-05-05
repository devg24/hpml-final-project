# HPML Final Project: [Project Title]

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** [Team Name]
- **Members:**
  - Dev Goyal (dg3513) — *role / area of contribution*
  - Anoushka Khajanachi (ak5446) — *role / area of contribution*
  - Alessandro Castillo ([UNI 3]) — *role / area of contribution*

## Submission

- **GitHub repository:** [https://github.com/devg24/hpml-final-project.git](https://github.com/devg24/hpml-final-project.git)
- **Final report:** [`deliverables/HPML_Final_Report.pdf`](deliverables/HPML_Final_Report.pdf)
- **Final presentation:** [`deliverables/HPML_Final_Presentation.pptx`](deliverables/HPML_Final_Presentation.pptx)
- **Experiment-tracking dashboard:** [https://wandb.ai/ak5446-columbia-university/hpml-final-project](https://wandb.ai/ak5446-columbia-university/hpml-final-project)

The final report PDF and the presentation file are checked into the `deliverables/` folder of this repository **and** uploaded to CourseWorks.

---

## 1. Problem Statement

This project evaluates and optimizes LLM inference for high-concurrency agentic workloads. We compare standard PyTorch backends against high-performance engines like vLLM and techniques such as NF4 quantization and speculative decoding to address compute and memory bandwidth bottlenecks in multi-agent systems.

---

## 2. Model/Application Description

Briefly describe the model(s) and stack you used:

- **Model architecture:** Qwen-7B-Instruct
- **Framework:** PyTorch 2.x, vLLM
- **Custom layers or modifications:** Prefix KV-cache sharing across agents.
- **Hardware target:** NVIDIA L4 (24GB VRAM)

---

## 3. Final Results Summary

All metrics measured at 50 concurrent agents on 1× NVIDIA L4 (24 GB),
CUDA 12.4, PyTorch 2.x, Qwen-7B-Instruct.

| Metric                        | HF bf16 (Baseline) | vLLM (Best)  | Δ                    |
|-------------------------------|--------------------|--------------|----------------------|
| Success rate @ 50 agents      | 8 / 50  (16%)      | 50 / 50 (100%) | +84 pp             |
| Throughput (tok/s) @ 50 agents| 1.2 tok/s          | 9.1 tok/s    | 7.6× higher          |
| TTFT p50 @ 1 agent            | 0.02 s             | 1.64 s       | +1.62 s overhead*    |
| OOM failures @ 50 agents      | 42                 | 0            | −42 crashes          |
| Peak VRAM @ 1 agent           | 19.24 GB           | vLLM-managed | —                    |
| Peak VRAM w/ NF4 @ 1 agent    | 19.24 GB           | 9.56 GB      | 50% less             |
| Wall time @ 50 agents         | 236 s              | 31.7 s       | 7.4× faster          |
| Max stable concurrency        | ~15 agents         | 400+ agents  | 26× more scalable    |

*vLLM TTFT is higher at low concurrency due to continuous batching scheduler
overhead. This inverts at 15+ agents where HF serialization dominates.

**Hardware:** 1× NVIDIA L4 (24 GB), CUDA 12.4, PyTorch 2.x, Google Cloud

**Headline result:** Replacing the native HuggingFace inference pipeline with
vLLM increased concurrent agent throughput 7.6× (1.2 → 9.1 tok/s at 50
agents), eliminated all OOM failures, and extended stable serving capacity
from ~15 agents to 400+ agents with zero crashes, on the same L4 GPU and
Qwen-7B-Instruct weights.

---

## 4. Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── agents_config.json      # Configuration for multi-agent personas
├── configs/                # YAML / JSON configs for every reported experiment
├── deliverables/           # Final report (PDF) and final presentation (PPT/PDF)
│   ├── HPML_Final_Report.pdf
│   └── HPML_Final_Presentation.pptx
├── scripts/
│   ├── run_benchmarks_final.sh
│   └── run_experiment.sh
├── src/
│   ├── backends/           # Inference backend implementations (vLLM, HF, etc.)
│   ├── contracts.py        # Data classes and interfaces
│   ├── baseline_agents.py  # Baseline agent definitions
│   ├── run_all.py          # Main orchestration and benchmarking entry point
│   ├── agent_maker.py      # Utility to manage agent configs
│   └── test_client.py      # Basic testing client
```

---

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
# Clone
git clone https://github.com/devg24/hpml-final-project.git
cd hpml-final-project

# (Recommended) create a clean Python environment
python -m venv .venv && source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

**System requirements:** Python 3.10+, CUDA 12.x, ≥ 24 GB GPU memory (e.g., NVIDIA L4 / 3090 / 4090).

### B. Experiment Tracking Dashboard

Public experiment-tracking dashboard with training and evaluation metrics, system profiling, and baseline vs. optimized comparisons:

> **🔗 Dashboard:** [https://wandb.ai/ak5446-columbia-university/hpml-final-project](https://wandb.ai/ak5446-columbia-university/hpml-final-project)
>
> *Platform used:* Weights & Biases

Verify the link opens in an incognito browser. The dashboard includes a curated **report** that walks through the optimization story. If your platform does not support public links (e.g., self-hosted MLflow), a static export is committed under `results/dashboard/` instead.

### C. Benchmarking
To reproduce the full suite of benchmarks:
```bash
bash scripts/run_benchmarks_final.sh
```

### D. Profiling
To regenerate Nsight Systems profiler traces:
```bash
bash scripts/run_benchmarks_final.sh  # Includes profiled runs in Part 5
```

### G. Quickstart: Reproduce the Headline Result

The following sequence reproduces the results end-to-end:

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Run all benchmarks (Baseline, NF4, Speculative, vLLM)
bash scripts/run_benchmarks_final.sh
```

---

## 6. Results and Observations

- **Baseline (HF bf16):** The native HuggingFace pipeline collapses under
  concurrency — throughput drops 16× from 14.3 tok/s (1 agent) to 0.9 tok/s
  (15 agents), and 42 of 50 agents crash with OOM at 50 agents. Nsight
  Systems confirms the cause: 1.56M `cudaLaunchKernel` calls (29.9% of API
  time) from PyTorch eager mode, and unfused SiLU/RMSNorm kernels consuming
  3.2% of GPU time as separate dispatches. CUDA kernel serialization means
  concurrent agents queue behind each other at the GPU level despite
  thread-level parallelism.

- **Optimization 1 — INT4 NF4 Quantization (HF + bitsandbytes):** VRAM
  halved at low concurrency (19.24 GB → 9.56 GB at 1 agent), enabling 0 OOM
  at 15 agents vs 4 failures in the baseline. However, throughput regressed
  55% at 1 agent (14.3 → 6.5 tok/s) and wall time exploded to 1005s at 15
  agents. Kernel analysis identifies the cause: `kgemm_4bit_inference_naive`
  has no Tensor Core path on the L4, so memory savings come at a severe
  compute cost. Quantization alone is insufficient for scalable serving.

- **Optimization 2 — Speculative Decoding (vLLM + Qwen-0.5B draft):**
  Improved single-agent throughput 9.0 tok/s vs 14.3 baseline, but performed
  worse under concurrency — only 6/15 agents succeeded at 15 agents and 3/50
  at 50 agents. The draft model occupies additional VRAM (~20 GB peak),
  leaving less headroom for KV cache growth. Speculative decoding benefits
  latency in low-concurrency settings but is counter-productive for
  high-concurrency serving on memory-constrained GPUs.

- **Optimization 3 — vLLM Engine (PagedAttention + continuous batching):**
  The dominant optimization. 50/50 agents succeeded with zero OOM at 50
  agents (vs 8/50 baseline), throughput degraded only 1.7× from 1 → 50
  agents (vs 16× for HF), and the system remained stable up to 400 concurrent
  agents. Kernel analysis shows vLLM eliminates unfused activation overhead
  entirely via `act_and_mul` and `fused_add_rms_norm`, replaces
  `CatArrayBatchedCopy` KV concatenation with in-place `reshape_and_cache_flash`
  block writes, and reduces kernel launch count 17× vs HF. At 400 agents,
  throughput settles at 3.7 tok/s — latency (TTFT 48.5 s) becomes the
  bottleneck, not memory.

- **Optimization 4 — vLLM + NF4 (combined):** Preserved all stability gains
  of vLLM (0 OOM up to 400 agents) while reducing model weight footprint.
  However, throughput degraded vs pure vLLM at every concurrency level
  (11.3 vs 15.2 tok/s at 1 agent; 2.9 vs 3.7 tok/s at 400 agents). Once
  vLLM manages KV cache efficiently, weight quantization adds overhead
  without meaningfully reducing the scheduling bottleneck.

- **What did not work:** NF4 quantization applied to the HF pipeline (not
  vLLM) worsened the throughput collapse rather than alleviating it — the
  dequantization overhead outweighed the VRAM savings under concurrent load.
  Speculative decoding failed at scale on a 24 GB GPU because the draft model
  itself consumes ~6 GB, leaving insufficient headroom for multi-agent KV
  caches. Both optimizations require a proper serving engine (vLLM) to be
  beneficial.

---

## 7. Notes

- Source files live under `src/`, configuration under `configs/`, and scripts under `scripts/`.
- All experiment logs and results are stored in `results/`.
- Wandb tokens and other secrets should be set as environment variables.

### AI Use Disclosure

**Did your team use any AI tool in completing this project?**

- [ ] No, we did not use any AI tool.
- [x] Yes, we used AI assistance as described below.

**Tool(s) used:** Claude, Gemini

**Specific purpose:** debugged environment issues, refactored project structure

**Sections affected:** README.md, scripts/run_benchmarks_final.sh, src/

**How we verified correctness:** manually verified script paths, ran help commands to check imports, and reran all experiments to confirm correctness.

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

If you build on this work, please cite:

```bibtex
@misc{teamname2026hpml,
  title  = {[Project Title]},
  author = {Last1, First1 and Last2, First2 and Last3, First3},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/devg24/hpml-final-project.git}
}
```

### Contact

Open a GitHub Issue or email *[team-contact@columbia.edu]*.

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*

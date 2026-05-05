#!/bin/bash

echo "Starting HPML Final Benchmarks..."

echo "Running task: Baseline with 15 agents"
nsys profile --trace=cuda,nvtx --output=trace_hf_baseline python src/run_all.py --backends hf_bf16 --n-agents 15

echo "Running task: Quantized with 15 agents"
nsys profile --trace=cuda,nvtx --output=trace_hf_quantized python src/run_all.py --backends hf_nf4 --n-agents 15

echo "Running task: Speculative with 15 agents"
nsys profile --trace=cuda,nvtx --output=trace_speculative python src/run_all.py --backends speculative --n-agents 15

echo "Running task: VLLM with 15 agents"
nsys profile --trace=cuda,nvtx --output=trace_vllm_15 python src/run_all.py --backends vllm --n-agents 15

echo "Running task: VLLM with 50 agents"
nsys profile --trace=cuda,nvtx --output=trace_vllm_50 python src/run_all.py --backends vllm --n-agents 50

echo "Running task: VLLM with 100 agents"
nsys profile --trace=cuda,nvtx --output=trace_vllm_100 python src/run_all.py --backends vllm --n-agents 100

echo "Running task: VLLM with 200 agents"
nsys profile --trace=cuda,nvtx --output=trace_vllm_200 python src/run_all.py --backends vllm --n-agents 200

echo "Running task: VLLM with 400 agents"
nsys profile --trace=cuda,nvtx --output=trace_vllm_400 python src/run_all.py --backends vllm --n-agents 400

echo "Running task: VLLM + quantized with 400 agents"
nsys profile --trace=cuda,nvtx --output=trace_vllm_bnb_nf4_400 python src/run_all.py --backends hf_vllm_bnb_nf4 --n-agents 400

echo "FINISHEDD YIPPEEEE XD"
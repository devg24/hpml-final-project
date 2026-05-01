#!/bin/bash

# Removed the 'e' so a Hugging Face crash doesn't kill the whole script!
set -uo pipefail

echo "=========================================="
echo "=========================================="

mkdir -p traces
mkdir -p logs
mkdir -p results

echo "Using Python:"
which python

echo "Using Nsight Systems:"
which nsys
nsys --version || true

############################################
# Shared Nsight Systems settings
############################################

# Changed to export=sqlite to generate .nsys-rep directly!
NSYS_PROFILE="nsys profile \
  --force-overwrite=true \
  --export=sqlite \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --stats=true"


############################################
# Part 1: Original profiled runs
############################################

echo "=========================================="
echo "PART 1: 5-Agent Runs"
echo "=========================================="

echo "Running task: Baseline  with 5 agents"
python run_all.py \
  --backends hf_bf16 \
  --n-agents 5

echo "Running task: HF NF4 quantized with 5 agents"
python run_all.py \
  --backends hf_nf4 \
  --n-agents 5

echo "Running task: Speculative with 5 agents" 
python run_all.py \
  --backends speculative \
  --n-agents 5

echo "Running task: vLLM with 5 agents"
python run_all.py \
  --backends vllm \
  --n-agents 5

echo "Running task: vLLM HF NF4 with 5 agents"
python run_all.py \
  --backends hf_vllm_bnb_nf4 \
  --n-agents 5

echo "=========================================="
echo "PART 1.5: 15-Agent Scaling (PROFILER OFF)"
echo "=========================================="


echo "Running task: baseline  with 15 agents"
python run_all.py \
  --backends hf_bf16 \
  --n-agents 15

echo "Running task: HF NF4 quantized with 15 agents"
python run_all.py \
  --backends hf_nf4 \
  --n-agents 15

echo "Running task: Speculative with 15 agents"
python run_all.py \
  --backends speculative \
  --n-agents 15

echo "Running task: vLLM with 15 agents"
python run_all.py \
  --backends vllm \
  --n-agents 15


############################################
# Part 2: vLLM scaling runs
############################################

echo "=========================================="
echo "PART 2: vLLM scaling runs (PROFILER OFF)"
echo "=========================================="

echo "Running task: vLLM with 50 agents"
python run_all.py \
  --backends vllm \
  --n-agents 50

echo "Running task: vLLM with 100 agents"
python run_all.py \
  --backends vllm \
  --n-agents 100

echo "Running task: vLLM with 200 agents"
python run_all.py \
  --backends vllm \
  --n-agents 200

echo "Running task: vLLM with 400 agents"
python run_all.py \
  --backends vllm \
  --n-agents 400

############################################
# Part 3: Additional stress runs
############################################

echo "=========================================="
echo "PART 3: Additional stress runs (PROFILER OFF)"
echo "=========================================="

echo "Running HF NF4 quantized with 50 agents"
python run_all.py \
  --backends hf_nf4 \
  --n-agents 50

echo "Running speculative decoding with 50 agents"
python run_all.py \
  --backends speculative \
  --n-agents 50

echo "Running HF vLLM BNB NF4 with 50 agents"
python run_all.py \
  --backends hf_vllm_bnb_nf4 \
  --n-agents 50

echo "Running HF vLLM BNB NF4 with 400 agents"
python run_all.py \
  --backends hf_vllm_bnb_nf4 \
  --n-agents 400


############################################
# Part 4: Main benchmark numbers
############################################

echo "=========================================="
echo "PART 4: Main benchmark numbers (PROFILER OFF)"
echo "=========================================="

echo "Running main 50-agent comparison"
python run_all.py \
  --backends hf_bf16 hf_nf4 speculative vllm hf_vllm_bnb_nf4 \
  --n-agents 50

echo "Running vLLM scaling numbers, no profiler"
python run_all.py \
  --backends vllm \
  --n-agents 15

python run_all.py \
  --backends vllm \
  --n-agents 100

python run_all.py \
  --backends vllm \
  --n-agents 200

python run_all.py \
  --backends vllm \
  --n-agents 400

echo "Running HF vLLM BNB NF4 scaling numbers, no profiler"
python run_all.py \
  --backends hf_vllm_bnb_nf4 \
  --n-agents 50

python run_all.py \
  --backends hf_vllm_bnb_nf4 \
  --n-agents 400

############################################
# Part 3: Single-agent kernel comparison
############################################

echo "=========================================="
echo "PART 5: Single-agent kernel comparison (PROFILED)"
echo "=========================================="

echo "Running HF BF16 with 1 agent"
$NSYS_PROFILE \
  --output=traces/hf_bf16_1agent \
  python run_all.py \
    --backends hf_bf16 \
    --n-agents 1

echo "Running HF NF4 with 1 agent"
$NSYS_PROFILE \
  --output=traces/hf_nf4_1agent \
  python run_all.py \
    --backends hf_nf4 \
    --n-agents 1

echo "Running vLLM with 1 agent"
$NSYS_PROFILE \
  --output=traces/vllm_1agent \
  python run_all.py \
    --backends vllm \
    --n-agents 1

echo "Running HF vLLM BNB NF4 with 1 agent"
$NSYS_PROFILE \
  --output=traces/hf_vllm_bnb_nf4_1agent \
  python run_all.py \
    --backends hf_vllm_bnb_nf4 \
    --n-agents 1



############################################
# Final file check
############################################

echo "=========================================="
echo "Generated nsys-rep files:"
echo "=========================================="
ls -lh traces/*.nsys-rep 2>/dev/null || echo "No .nsys-rep files found."

echo "=========================================="
echo "All HPML final benchmark runs completed."
echo "FINISHED YIPPEEEE XD"
echo "=========================================="
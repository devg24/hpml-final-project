import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.streamers import BaseStreamer
import asyncio
import time
import wandb
import traceback
import json
import os
import inspect
import argparse
import torch.nn.modules.module as module

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--optim", type=str, default="none", choices=["none", "quantization"],
                    help="Optimization to apply: 'none' (default bfloat16) or 'quantization' (INT4 NF4)")
args = parser.parse_args()

# --- Configuration ---
MODEL_PATH = "./qwen-7b"
WANDB_PROJECT = "hpml-final-project"
CONCURRENT_AGENTS = 15
MAX_NEW_TOKENS = 256
DEVICE = "cuda:0"
# MAX_INPUT_TOKENS = 1020     # <-- Controls max INPUT length (Prompt + Code)


# --- File Paths ---
AGENTS_FILE = "agents_config.json"

# --- Load External Data ---
print(f"Loading workload from {AGENTS_FILE}...")

if not os.path.exists(AGENTS_FILE):
    raise FileNotFoundError("Ensure 'agents_config.json' exists in the directory.")

SHARED_CODE_PREFIX = inspect.getsource(module)
SHARED_CODE_PREFIX = "\n".join(SHARED_CODE_PREFIX.splitlines()[:600])

with open(AGENTS_FILE, "r") as f:
    ALL_AGENTS = json.load(f).get("agents", [])

print(f"Loaded prefix length: {len(SHARED_CODE_PREFIX)} characters.")
print(f"Loaded {len(ALL_AGENTS)} agent profiles.")

# --- Custom Streamer for TTFT ---
class TTFTStreamer(BaseStreamer):
    """Custom streamer to capture the exact time the first token is generated."""
    def __init__(self):
        self.start_time = time.time()
        self.first_token_time = None
        self.token_count = 0

    def put(self, value):
        if self.first_token_time is None:
            self.first_token_time = time.time()
        self.token_count += 1

    def end(self):
        pass

# --- Model Initialization ---
print(f"Loading Model and Tokenizer into VRAM (optim={args.optim})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

if args.optim == "quantization":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NF4 format — fused kernel, no dequant buffer
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,      # quantize the quantization constants too
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=DEVICE,
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
model.eval()
print("Model loaded successfully.")

# --- OOM-Safe Perplexity Helper ---
def calculate_perplexity(model, input_ids, labels=None, max_chunk_size=2048):
    """Calculates perplexity, chunking long sequences to prevent OOM."""
    seq_len = input_ids.size(1)
    
    with torch.no_grad():
        # Fast path for short sequences
        if seq_len <= max_chunk_size:
            outputs = model(input_ids, labels=labels if labels is not None else input_ids)
            return torch.exp(outputs.loss).item()
            
        # Safe chunking path for massive sequences
        nlls = []
        valid_tokens = 0
        
        for i in range(0, seq_len, max_chunk_size):
            chunk_inputs = input_ids[:, i:i + max_chunk_size]
            chunk_labels = labels[:, i:i + max_chunk_size] if labels is not None else chunk_inputs
            
            outputs = model(chunk_inputs, labels=chunk_labels)
            
            active_tokens = (chunk_labels != -100).sum().item()
            if active_tokens > 0:
                nlls.append(outputs.loss.item() * active_tokens)
                valid_tokens += active_tokens
        
        if valid_tokens == 0:
            return 0.0
            
        avg_loss = sum(nlls) / valid_tokens
        return torch.exp(torch.tensor(avg_loss)).item()

# --- Generation Function (Runs in Thread) ---
def generate_review(agent_id: int, task_prompt: str):
    """Synchronous generation function to be executed concurrently."""
    full_prompt = f"Code:\n{SHARED_CODE_PREFIX}\n\nSystem: {task_prompt}\n\nReview:"
    
    # NEW: Apply the input token limit with truncation
    inputs = tokenizer(
        full_prompt, 
        return_tensors="pt"
        # truncation=True,                # <-- Turn on truncation
        # max_length=MAX_INPUT_TOKENS      # <-- Apply the limit here
    ).to(DEVICE)

    streamer = TTFTStreamer()
    start_time = time.time()

    cuda_stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(cuda_stream):
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS, # <-- Output limit applied here
                streamer=streamer,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        total_time = end_time - start_time
        ttft = streamer.first_token_time - start_time if streamer.first_token_time else total_time
        
        input_len = inputs["input_ids"].shape[-1]
        output_tokens = len(outputs[0]) - input_len
        throughput = output_tokens / total_time
        
        # Ship the tensor to the CPU immediately to free up GPU VRAM!
        full_ids_cpu = outputs[0].unsqueeze(0).cpu()
        
        return {
            "agent_id": agent_id,
            "status": "success",
            "ttft_seconds": ttft,
            "throughput_tps": throughput,
            "total_time_seconds": total_time,
            "output_tokens": output_tokens,
            "full_ids_cpu": full_ids_cpu, 
            "input_len": input_len        
        }
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "agent_id": agent_id,
            "status": "OOM_ERROR",
            "error_trace": traceback.format_exc()
        }
    except Exception as e:
        return {
            "agent_id": agent_id,
            "status": "ERROR",
            "error_trace": str(e)
        }

# --- Async Coordinator ---
async def main():
    actual_agents_to_run = min(CONCURRENT_AGENTS, len(ALL_AGENTS))
    
    wandb.init(
        entity="ak5446-columbia-university",
        project=WANDB_PROJECT,
        name=f"baseline_{args.optim}_hf_{actual_agents_to_run}_agents",
        config={
            "model": "Qwen-7B-Instruct",
            "framework": "HuggingFace-PyTorch",
            "optim": args.optim,
            "concurrent_agents": actual_agents_to_run,
            "prefix_length_approx": len(SHARED_CODE_PREFIX),
            "max_new_tokens": MAX_NEW_TOKENS
        }
    )
    
    print(f"Starting simulation with {actual_agents_to_run} concurrent agents...")
    agents_to_run = ALL_AGENTS[:actual_agents_to_run]
    
    tasks = [
        asyncio.to_thread(generate_review, agent["id"], agent["persona"]) 
        for agent in agents_to_run
    ]
    
    # Calculate static Base Code Perplexity
    print("Calculating Base Code Perplexity...")
    base_code_inputs = tokenizer(SHARED_CODE_PREFIX, return_tensors="pt").to(DEVICE)
    base_ppl = calculate_perplexity(model, base_code_inputs["input_ids"])
    print(f"Base Code Perplexity: {base_ppl:.4f}")

    results = await asyncio.gather(*tasks)
    
    # Sequential Perplexity Calculation
    print("\nCalculating Perplexity for successful generations sequentially...")
    for res in results:
        if res["status"] == "success":
            full_ids = res["full_ids_cpu"].to(DEVICE)
            input_len = res["input_len"]
            
            labels = full_ids.clone()
            labels[:, :input_len] = -100
            
            res["perplexity"] = calculate_perplexity(model, full_ids, labels=labels)
            
            # Nuke tensors from GPU
            del full_ids
            del labels
            torch.cuda.empty_cache() 
            
            print(f"Agent {res['agent_id']} Perplexity: {res['perplexity']:.4f}")
            del res["full_ids_cpu"] 
            
    # Process and log results
    successful_runs = 0
    oom_crashes = 0
    avg_ttft = 0
    avg_throughput = 0
    
    for res in results:
        if res["status"] == "success":
            successful_runs += 1
            avg_ttft += res["ttft_seconds"]
            avg_throughput += res["throughput_tps"]
            avg_ppl = res.get("perplexity", 0)
            wandb.log({
                f"agent_{res['agent_id']}/ttft": res["ttft_seconds"],
                f"agent_{res['agent_id']}/throughput": res["throughput_tps"],
                f"agent_{res['agent_id']}/perplexity": avg_ppl
            })
        elif res["status"] == "OOM_ERROR":
            oom_crashes += 1
            print(f"Agent {res['agent_id']} failed with OOM.")
            
    if successful_runs > 0:
        avg_ttft /= successful_runs
        avg_throughput /= successful_runs
        
    print("\n--- Simulation Complete ---")
    print(f"Successful Runs: {successful_runs}/{actual_agents_to_run}")
    print(f"OOM Crashes: {oom_crashes}/{actual_agents_to_run}")
    if successful_runs > 0:
        print(f"Average TTFT: {avg_ttft:.2f}s")
        print(f"Average Throughput: {avg_throughput:.2f} tokens/s")

    total_avg_ppl = 0
    if successful_runs > 0:
        total_avg_ppl = sum(res["perplexity"] for res in results if res["status"] == "success") / successful_runs

    wandb.log({
        "aggregate/success_rate": successful_runs / actual_agents_to_run,
        "aggregate/oom_count": oom_crashes,
        "aggregate/avg_ttft": avg_ttft,
        "aggregate/avg_throughput": avg_throughput,
        "aggregate/avg_perplexity": total_avg_ppl,
        "aggregate/base_code_perplexity": base_ppl
    })
    
    wandb.finish()

if __name__ == "__main__":
    asyncio.run(main())
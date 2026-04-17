import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
import asyncio
import time
import wandb
import traceback
import json
import os
import torch.nn.functional as F
import inspect
import numpy as np

# --- Configuration ---
MODEL_PATH = "./qwen-7b" 
WANDB_PROJECT = "hpml-final-project"
CONCURRENT_AGENTS = 5 # Scale to 15, 30, 50 to force OOM
MAX_NEW_TOKENS = 256
DEVICE = "cuda:0" # <-- Added explicit device

# --- File Paths ---
AGENTS_FILE = "agents_config.json"

# --- Load External Data ---
print(f"Loading workload from {AGENTS_FILE}...")

if not os.path.exists(AGENTS_FILE):
    raise FileNotFoundError("Ensure 'agents_config.json' exists in the directory.")

SHARED_CODE_PREFIX = inspect.getsource(module) 

with open(AGENTS_FILE, "r") as f:
    ALL_AGENTS = json.load(f).get("agents", [])
print(SHARED_CODE_PREFIX[:500], "\n...\n")  # Print the first 500 characters of the prefix for verification
print(ALL_AGENTS[:2], "\n...\n")  # Print the first 2 agent profiles for verification

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
print("Loading Model and Tokenizer into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE, # <-- Changed from "auto" to DEVICE
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully.")

# --- Perplexity Helper ---
def calculate_perplexity(model, input_ids, labels=None):
    """Calculates perplexity for given input_ids, optionally masking the prompt."""
    with torch.no_grad():
        # If labels are provided, loss is only calculated on non-masked tokens (-100)
        outputs = model(input_ids, labels=labels if labels is not None else input_ids)
        loss = outputs.loss
        return torch.exp(loss).item()

# --- Generation Function (Runs in Thread) ---
def generate_review(agent_id: int, task_prompt: str):
    """Synchronous generation function to be executed concurrently."""
    full_prompt = f"System: {task_prompt}\n\nCode:\n{SHARED_CODE_PREFIX}\n\nReview:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE) # <-- Changed from "cuda" to DEVICE
    
    streamer = TTFTStreamer()
    start_time = time.time()
    
    try:
        # Naive synchronous generation
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            streamer=streamer,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        end_time = time.time()
        
        # Calculate Metrics
        total_time = end_time - start_time
        ttft = streamer.first_token_time - start_time if streamer.first_token_time else total_time
        
        input_len = inputs["input_ids"].shape[-1]
        output_tokens = len(outputs[0]) - input_len
        throughput = output_tokens / total_time
        
        # Calculate Generation Perplexity (mask the prompt)
        full_ids = outputs[0].unsqueeze(0)
        labels = full_ids.clone()
        labels[:, :input_len] = -100
        gen_ppl = calculate_perplexity(model, full_ids, labels=labels)
        
        return {
            "agent_id": agent_id,
            "status": "success",
            "ttft_seconds": ttft,
            "throughput_tps": throughput,
            "total_time_seconds": total_time,
            "output_tokens": output_tokens,
            "perplexity": gen_ppl
        }
        
    except torch.cuda.OutOfMemoryError:
        # Catch OOM to prevent the entire script from silently crashing
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
    # Ensure we don't request more agents than we have configured
    actual_agents_to_run = min(CONCURRENT_AGENTS, len(ALL_AGENTS))
    
    wandb.init(
        entity="ak5446-columbia-university",
        project=WANDB_PROJECT,
        name=f"baseline_naive_hf_{actual_agents_to_run}_agents",
        config={
            "model": "Qwen-7B-Instruct",
            "framework": "HuggingFace-PyTorch",
            "concurrent_agents": actual_agents_to_run,
            "prefix_length_approx": len(SHARED_CODE_PREFIX),
            "max_new_tokens": MAX_NEW_TOKENS
        }
    )
    
    print(f"Starting simulation with {actual_agents_to_run} concurrent agents...")
    
    # Slice the loaded JSON list
    agents_to_run = ALL_AGENTS[:actual_agents_to_run]
    
    # Wrap the synchronous generate function in asyncio.to_thread
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

    # Log aggregate metrics to WandB
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
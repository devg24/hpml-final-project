import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BaseStreamer
import asyncio
import time
import wandb
import traceback

# --- Configuration ---
# Update this to point to your local symlink in the VM (e.g., "./qwen-7b-model")
MODEL_PATH = "./qwen-7b" 
WANDB_PROJECT = "hpml-final-project"
CONCURRENT_AGENTS = 5 # Start small (e.g., 5), then scale to 15, 30, 50 to force OOM
MAX_NEW_TOKENS = 256

# --- Mock Data ---
# A dummy 2,000-line equivalent code prefix
SHARED_CODE_PREFIX = "def example_function():\n    pass\n" * 2000

# Distinct agent personas (The "Unique Suffixes")
AGENT_TASKS = [
    "Review the above code for security vulnerabilities.",
    "Review the above code for PEP8 style compliance and readability.",
    "Review the above code for time and space complexity optimizations.",
    "Generate comprehensive docstrings for the above code.",
    "Identify any potential race conditions or concurrency issues in the above code."
] * 10 # Multiply to allow up to 50 concurrent agents

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
    device_map="auto", # Automatically places it on the L4 GPU
    torch_dtype=torch.bfloat16, # Crucial for fitting 7B on 24GB VRAM
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully.")

# --- Generation Function (Runs in Thread) ---
def generate_review(agent_id: int, task_prompt: str):
    """Synchronous generation function to be executed concurrently."""
    full_prompt = f"System: {task_prompt}\n\nCode:\n{SHARED_CODE_PREFIX}\n\nReview:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
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
        output_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
        throughput = output_tokens / total_time
        
        return {
            "agent_id": agent_id,
            "status": "success",
            "ttft_seconds": ttft,
            "throughput_tps": throughput,
            "total_time_seconds": total_time,
            "output_tokens": output_tokens
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
    wandb.init(
        project=WANDB_PROJECT,
        name=f"baseline_naive_hf_{CONCURRENT_AGENTS}_agents",
        config={
            "model": "Qwen-7B-Instruct",
            "framework": "HuggingFace-PyTorch",
            "concurrent_agents": CONCURRENT_AGENTS,
            "prefix_length_approx": len(SHARED_CODE_PREFIX),
            "max_new_tokens": MAX_NEW_TOKENS
        }
    )
    
    print(f"Starting simulation with {CONCURRENT_AGENTS} concurrent agents...")
    
    # Select the required number of agent tasks
    tasks_to_run = AGENT_TASKS[:CONCURRENT_AGENTS]
    
    # Wrap the synchronous generate function in asyncio.to_thread to simulate concurrency.
    # Note: PyTorch releases the GIL during C++ execution, so these WILL hit the GPU concurrently,
    # causing the massive memory fragmentation/OOM you want to document.
    tasks = [
        asyncio.to_thread(generate_review, i, task_prompt) 
        for i, task_prompt in enumerate(tasks_to_run)
    ]
    
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
            wandb.log({
                f"agent_{res['agent_id']}/ttft": res["ttft_seconds"],
                f"agent_{res['agent_id']}/throughput": res["throughput_tps"]
            })
        elif res["status"] == "OOM_ERROR":
            oom_crashes += 1
            print(f"Agent {res['agent_id']} failed with OOM.")
            
    if successful_runs > 0:
        avg_ttft /= successful_runs
        avg_throughput /= successful_runs
        
    print("\n--- Simulation Complete ---")
    print(f"Successful Runs: {successful_runs}/{CONCURRENT_AGENTS}")
    print(f"OOM Crashes: {oom_crashes}/{CONCURRENT_AGENTS}")
    if successful_runs > 0:
        print(f"Average TTFT: {avg_ttft:.2f}s")
        print(f"Average Throughput: {avg_throughput:.2f} tokens/s")

    # Log aggregate metrics to WandB
    wandb.log({
        "aggregate/success_rate": successful_runs / CONCURRENT_AGENTS,
        "aggregate/oom_count": oom_crashes,
        "aggregate/avg_ttft": avg_ttft,
        "aggregate/avg_throughput": avg_throughput
    })
    
    wandb.finish()

if __name__ == "__main__":
    asyncio.run(main())
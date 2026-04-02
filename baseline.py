import time
import json
import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 1. Custom Streamer to catch Time-To-First-Token (TTFT)
class ProfilerStreamer(TextStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, **kwargs)
        self.start_time = None
        self.first_token_time = None
        self.token_count = 0

    def put(self, value):
        if self.first_token_time is None:
            self.first_token_time = time.time()
        
        # HuggingFace sometimes passes multi-dimensional tensors, ensure we count correctly
        self.token_count += value.numel() if isinstance(value, torch.Tensor) else 1
        super().put(value)

# 2. Initialize WandB
wandb.init(
    project="hpml-code-swarm",
    name="baseline-pytorch-sync",
    config={
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "framework": "HuggingFace-PyTorch",
        "max_new_tokens": 256,
        "hardware": "NVIDIA L4 24GB",
        "precision": "bfloat16"
    }
)

# 3. Load Model & Tokenizer
model_id = wandb.config.model_name
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

# 4. Load Dataset
with open("shared_prefix.py", "r") as f:
    shared_code = f.read()

with open("agents_config.json", "r") as f:
    agents = json.load(f) # List of dicts: [{"id": 1, "persona": "..."}]

# 5. The Baseline Execution Loop
for agent in agents:
    print(f"--- Running Agent {agent['id']}: {agent['persona'][:30]}... ---")
    
    # Construct the prompt
    messages = [
        {"role": "system", "content": agent['persona']},
        {"role": "user", "content": f"Review this code:\n\n{shared_code}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    # Setup Streamer
    streamer = ProfilerStreamer(tokenizer)
    streamer.start_time = time.time()
    
    try:
        # Generate
        outputs = model.generate(
            **inputs, 
            max_new_tokens=wandb.config.max_new_tokens,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Calculate Metrics
        end_time = time.time()
        ttft = streamer.first_token_time - streamer.start_time
        decode_time = end_time - streamer.first_token_time
        
        # Throughput: Generated tokens / decode time
        # Subtract prompt length from total output length to get generated tokens
        generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        throughput = generated_tokens / decode_time if decode_time > 0 else 0
        
        # 6. Log to WandB
        wandb.log({
            "agent_id": agent['id'],
            "ttft_seconds": ttft,
            "decode_time_seconds": decode_time,
            "throughput_tok_sec": throughput,
            "total_latency_seconds": end_time - streamer.start_time,
            "gpu_memory_allocated_GB": torch.cuda.memory_allocated() / (1024**3),
            "gpu_memory_reserved_GB": torch.cuda.memory_reserved() / (1024**3)
        })
        
    except torch.cuda.OutOfMemoryError:
        print(f"OOM Error on Agent {agent['id']}!")
        wandb.log({"OOM_crash": 1, "crashed_on_agent": agent['id']})
        break # Exit loop on crash

wandb.finish()
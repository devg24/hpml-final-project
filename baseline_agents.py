import asyncio
from sglang import RuntimeEndpoint, gen

# This represents your 5-week project's "Shared Context"
SYSTEM_PROMPT = "You are a specialized agent in a high-performance simulation. " * 100 # Simulated long prompt

async def run_agent(endpoint, agent_id):
    state = endpoint.run(
        f"{SYSTEM_PROMPT}\nAgent {agent_id}, analyze the current GPU state.",
        temperature=0.0
    )
    print(f"Agent {agent_id} Response: {state['text'][:50]}...")

async def main():
    # 1. Initialize the SGLang Runtime (Backend)
    # This automatically uses RadixAttention (Optimization #1)
    runtime = RuntimeEndpoint("http://localhost:30000")
    
    # 2. Run 5 agents in parallel
    print("Launching Agents...")
    await asyncio.gather(*(run_agent(runtime, i) for i in range(5)))

if __name__ == "__main__":
    asyncio.run(main())
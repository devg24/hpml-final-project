import openai

# Connect to your local SGLang server
client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

def chat_with_agent(prompt):
    response = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content

print("Response:", chat_with_agent("Explain why GPU memory layout matters for multi-agent systems."))
import json

file_path = "agents_config.json"  # change this to your JSON file name

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

for i, agent in enumerate(data["agents"], start=1):
    agent["id"] = i

with open(file_path, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=2)

print(f"Fixed agent IDs and rewrote {file_path}")
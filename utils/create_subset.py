import json

with open("data/prompts/prompts.json") as f:
    data = json.load(f)

subset = data[:1000]

with open("data/prompts/prompts_subset.json", "w") as f:
    json.dump(subset, f, indent=2)
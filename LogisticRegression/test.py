from datasets import load_dataset
load_dataset("camel-ai/biology", split="train").shuffle(seed=42)


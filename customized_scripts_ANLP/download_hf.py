from datasets import load_dataset
import pandas as pd

DATASET_NAME = "SEACrowd/myxnli"
dataset = load_dataset(DATASET_NAME, split="train",trust_remote_code=True)
df = dataset.to_pandas()

df.to_csv("myxnli", index=False)

print(f"Dataset saved")
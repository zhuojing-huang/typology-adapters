from datasets import load_dataset
import random
import json

# Load the Oscar Chinese dataset in streaming mode
dataset = load_dataset("oscar-corpus/OSCAR-2201",
                       use_auth_token="...", #replace with your token
                       language="zh",  ##adjust for downloading different languages 
                       streaming=True,
                       split="train",
                       trust_remote_code=True)

# Collect the first 100,000 examples for high resource languages
samples = []
for i, sample in enumerate(dataset):
    if i >= 100000:   
        break
    samples.append(sample)

# Shuffle 
random.shuffle(samples)

# 80% train, 20% validation
split_index = int(0.8 * len(samples))
train_samples = samples[:split_index]
validation_samples = samples[split_index:]

# save as json
train_file = r"C:\Users\zoehu\OneDrive - stud.uni-goettingen.de\CS\Advanced NLP\wuu_train_80pct.json"
valid_file = r"C:\Users\zoehu\OneDrive - stud.uni-goettingen.de\CS\Advanced NLP\wuu_valid_20pct.json"

with open(train_file, "w", encoding="utf-8") as f:
    json.dump(train_samples, f, ensure_ascii=False, indent=4)

with open(valid_file, "w", encoding="utf-8") as f:
    json.dump(validation_samples, f, ensure_ascii=False, indent=4)

print("Saved")

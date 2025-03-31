import pandas as pd
import torch
from tqdm import tqdm
from transformers import XLMRobertaTokenizer
from adapters import XLMRobertaAdapterModel, AdapterConfig
import adapters.composition as ac
from sklearn.metrics import accuracy_score
import torch.nn.init as init

# load the base model
model_name = "xlm-roberta-base"
model = XLMRobertaAdapterModel.from_pretrained(model_name)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

# load trained adapters
adapter1_path = "C:/Users/zoehu/OneDrive - stud.uni-goettingen.de/CS/Advanced NLP/output_my_bo_zh/mlm"
adapter2_path = "C:/Users/zoehu/OneDrive - stud.uni-goettingen.de/CS/Advanced NLP/output_my_bo/mlm"
adapter3_path = "C:/Users/zoehu/OneDrive - stud.uni-goettingen.de/CS/Advanced NLP/output_my/mlm"
adapter4_path = "C:/Users/zoehu/OneDrive - stud.uni-goettingen.de/CS/Advanced NLP/glue_mnli/mnli"

adapter1 = model.load_adapter(adapter1_path, load_as="mlm_adapter1")
adapter2 = model.load_adapter(adapter2_path, load_as="mlm_adapter2")
adapter3 = model.load_adapter(adapter3_path, load_as="mlm_adapter3")
adapter4 = model.load_adapter(adapter4_path)
#model.add_adapter("pfeiffer_adapter", config="pfeiffer") #random layer; only activate when setting random baseline

#config = AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
#adp_swahili=model.load_adapter("sw/wiki@ukp", config=config)  #random ablation for irrelavent language 


# Stack adapters 
model.active_adapters = ac.Stack(adapter4)
#model.add_classification_head("mnli", num_labels=3,overwrite_ok=True) #random untrained classification head; only activate when setting random baseline

csv_path = "C:/Users/zoehu/OneDrive - stud.uni-goettingen.de/CS/Advanced NLP/testdata/myxnli_adjusted.csv"
df = pd.read_csv(csv_path)
df = df.head(100)

# Ensure the label column is integer
df['label'] = df['label'].astype(int)
print(df['label'].value_counts(normalize=True))

# label mapping
label_mapping = ["entailment", "neutral", "contradiction"]

y_true = []
y_pred = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing", unit="sample"):
    premise = row['sentence1_my']
    hypothesis = row['sentence2_my']
    true_label = row['label']

    inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probabilities).item()
    
    y_true.append(true_label)
    y_pred.append(predicted_label)

    # print some inference samples 
    if idx < 5:
        print("/n--- Inference Sample ---")
        print(f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"True Label: {label_mapping[true_label]}")
        print(f"Predicted Label: {label_mapping[predicted_label]}")
        print(f"Probabilities: {probabilities.numpy()}/n")

# accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

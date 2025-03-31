import pandas as pd
import torch
from tqdm import tqdm
from transformers import XLMRobertaTokenizer
from adapters import XLMRobertaAdapterModel, AdapterConfig
import adapters.composition as ac
from sklearn.metrics import accuracy_score
import torch.nn.init as init

# Load the base model
model_name = "xlm-roberta-base"
model = XLMRobertaAdapterModel.from_pretrained(model_name)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

#config = AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
#adp_swahili=model.load_adapter("sw/wiki@ukp", config=config)  #random ablation for unrelated language 

# Load trained adapters
adapter1_path = "C:/Users/zoehu/OneDrive - stud.uni-goettingen.de/CS/Advanced NLP/output_th_lo+km+vi/mlm"
adapter2_path = "C:/Users/zoehu/OneDrive - stud.uni-goettingen.de/CS/Advanced NLP/output_th_lo_my/mlm"
adapter3_path = "C:/Users/zoehu/OneDrive - stud.uni-goettingen.de/CS/Advanced NLP/output_my/mlm"
adapter4_path = "C:/Users/zoehu/OneDrive - stud.uni-goettingen.de/CS/Advanced NLP/glue/mrpc"

# Load the adapters
adapter1 = model.load_adapter(adapter1_path,load_as="mlm_adapter1")
adapter2 = model.load_adapter(adapter2_path,load_as="mlm_adapter2")
adapter3 = model.load_adapter(adapter3_path,load_as="mlm_adapter3")
adapter4 = model.load_adapter(adapter4_path)
#model.add_adapter("pfeiffer_adapter", config="pfeiffer") #random layer; activate only when setting random baseline

# Stack adapters (for multi-task learning)
model.active_adapters = ac.Stack("mlm_adapter1","mlm_adapter2","mlm_adapter3",adapter4) # stacking adapters; adjust every time for different setup
#model.add_classification_head("mrpc", num_labels=2) #random classification head

### Inference ###
csv_path = r"C:\Users\zoehu\OneDrive - stud.uni-goettingen.de\CS\Advanced NLP\testdata\paraphrase_detection\open-test.final.manual.csv"
df = pd.read_csv(csv_path)
df = df.head(500)

# ensure the label column is integer
df.iloc[:, 3] = df.iloc[:, 3].astype(int)
print(df.iloc[:, 3].value_counts(normalize=True))

# define label mapping 
label_mapping = ["not equivalent", "equivalent"]

y_true = []
y_pred = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing", unit="sample"):
    text1 = row.iloc[1]
    text2 = row.iloc[2]
    true_label = row.iloc[3]

    inputs = tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # convert logits to probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probabilities).item()
    
    y_true.append(true_label)
    y_pred.append(predicted_label)

    # print some inference samples 
    if idx < 5:
        print("\n--- Inference Sample ---")
        print(f"Premise: {text1}")
        print(f"Hypothesis: {text2}")
        print(f"True Label: {label_mapping[true_label]}")
        print(f"Predicted Label: {label_mapping[predicted_label]}")
        print(f"Probabilities: {probabilities.numpy()}\n")

# accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

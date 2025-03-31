<!---
Copyright 2020 The AdapterHub Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


## Phylogeny-Inspired and Contact-Based Adaptation: An Example of Sino-Tibetan Languages and SEA Sprachbund

- This is a seminar project for Advanced Topics for NLP.
- The repository is forked from the original adapter repository.
- Only the folder `customized_scripts_ANLP` has custonmized code for the project.

### Overview
This project investigates the effectiveness of language adapter stacking for two NLP tasks: Natural Language Inference (NLI) and Paraphrase Classification. The focus is on Southeast Asian languages and Swahili, with experiments conducted on different adapter training configurations. The primary objective is to determine whether stacking adapters based on linguistic phylogeny or contact relationships enhances downstream task performance.



### Languages and Tasks
The study includes languages from four language families:

| Family         | Genus         | Languages      |
|--------------|--------------|-----------------|
| Sino-Tibetan | Sinitic      | Chinese (zh)    |
|              | Tibeto-Burman | Tibetan (bo), Burmese (my) |
| Kra-Dai      | Tai          | Thai (th), Lao (lo) |
| Austroasiatic | Mon-Khmer   | Vietnamese (vi), Khmer (km) |
| Bantu        | Sabaki      | Swahili (sw) |

The project evaluates two downstream tasks on Burmese:
- **Natural Language Inference (NLI)**
- **Paraphrase Classification**

### Datasets
The following datasets are used for training and evaluation:
- **Pre-training Dataset**: OSCAR Corpus (15,000–100,000 sentences per language)
- **Evaluation Datasets**:
  - **NLI**: MyXNLI dataset [![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-orange?logo=huggingface)](https://huggingface.co/datasets/SEACrowd/myxnli)
  - **Paraphrase Classification**: my_paraphrase dataset [![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-orange?logo=huggingface)](https://huggingface.co/datasets/SEACrowd/my_paraphrase)

Performance is measured using **accuracy** on a sample of 500 test instances per task.

### Adapter Training 
Language adapters are trained using:
- **Model**: XLM-RoBERTa Base
- **Framework**: AdapterHub
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Epochs**: 3
- **Configuration**: "seq_bn_inv" (invertible adapter from MAD-X)

### Baselines and Model Variations
The following configurations are tested:
- **Random Baseline**: Randomly initialized adapter with untrained classification head.
- **Task Adapter Only**: Task-specific adapter trained on English data (MNLI for NLI, MRPC for Paraphrase Classification).
- **Mono-Lingual Adapters**: Single language-specific adapters.
- **Multi-Lingual Adapters**: Cross-lingual adapters combining multiple languages.
- **Phylogeny-Based Cross-Lingual Adapters**: Stacking adapters based on linguistic family hierarchy.
- **Contact-Based Cross-Lingual Adapters**: Stacking adapters based on high lexical similarity due to language contact.

### Workflow

0. **Installation**

`adapters` currently supports **Python 3.9+** and **PyTorch 2.0+**.
After [installing PyTorch](https://pytorch.org/get-started/locally/), you can install `adapters` from PyPI ...

```
pip install -U adapters
```

... or from source by cloning the repository:

```
git clone https://github.com/adapter-hub/adapters.git
cd adapters
pip install .
```

1. **Download OSCAR dataset for training adapters**
- Run `customized_scripts_ANLP/download_oscar`. 
- Adjust "language" to desired languages.
- Before running, an authentification token should be given. 


2. **Pre-training Adapters**
- Train language adapters using the OSCAR dataset with the AdapterHub framework.
- If using SLURM framework, submit the job by using `customized_scripts_ANLP/train_language_adp.sh`.
- `customized_scripts_ANLP/train_language_adp.sh` submits `customized_scripts_ANLP/run_mlm.py`.
   
2. **Training Task Adapters**
- Train task adapters on the MNLI and MRPC datasets.
- Submit `customized_scripts_ANLP/train_glue.sh` to your HPC, which essentially runs `customized_scripts_ANLP/run_glue.py`.

3. **Adapter Stacking for Inference**
- Experiment with different stacking configurations as described above.

4. **Evaluation**
- Evaluate models using MyXNLI and my_paraphrase datasets.
- Example code for downloading HuggingFace dataset is available at `customized_scripts_ANLP/download_hf.py`
- To measure performance with accuracy, run `customized_scripts_ANLP/inference_my_nli.py` for NLI and run `customized_scripts_ANLP/inference_my_pc.py` for Paraphrase Classification.

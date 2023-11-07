# Evaluating Linguistic Style Adversarial Paraphrase Robustness of LLMs in Social and Commonsense QA

Large language models (LLMs) have made significant strides in natural language processing (NLP) tasks, but their performance needs to be assessed for linguistic variations and adversarial paraphrases. 
Previous works like HELM have laid the foundation for this research, but they primarily focus on local robustness (fixed transformations, spelling impurities etc.) 
We aim to take a broader approach, introducing global adversarial paraphrasing, which changes the style and structure of the input sentences while preserving the meaning and context.

## How we do it
Given that LLMs are effective in capturing context, we use them for paraphrasing the input sentences in different styles based on demographic groups of age gender and temporal changes. We then evaluate the performance of LLMs on social and commonsense question answering (QA) tasks, using the original and paraphrased sentences as inputs. We also compare the results with baselines.

## What this repository contains
This repository contains the following files and folders:

- `datasets`: 
- `llm-benchmark-notebooks`
- `llm-paraphrase-notebooks`
- `datasets`
- `baselines`
- `tempobert`
- `BIG-bench`

## Running the code


```bash
git clone https://github.com/llm-robustness/llm-robustness.git
cd llm-robustness
pip install -r requirements.txt
```

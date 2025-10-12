# Aligning Large Language Models with Preference Privacy

This repository contains code implementing the methods described in the paper:

> **[Aligning Large Language Models with Preference Privacy]**

---

## 📘 Overview

This project explores **preference-level privacy** in aligning large language models (LLMs) using human feedback. It implements:

- **Randomized Response (RR)**: Adds label-level noise to ensure differential privacy for human preferences.
- **DP-Stochastic Gradient Descent (DP-SGD)**: Provides full-data differential privacy, used for comparison.
- **PROPS (Progressively Private Self-Alignment)**: A multi-stage self-alignment strategy that refines noisy labels using intermediate models trained under privacy constraints.

---

## 🔧 Features

- ✅ Privacy-preserving training using **label-only differential privacy** (via RR).
- ✅ Multi-stage alignment with **PROPS**, improving performance under tight privacy budgets.
- ✅ Tools for comparing **privacy–utility trade-offs** across alignment strategies.
- ✅ Compatible with Hugging Face Transformers.

---

## 📊 Datasets Used

PROPS has been evaluated using the following **three preference datasets**:

1. **[AlpacaEval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval)** – Benchmark for evaluating LLMs with preference comparisons.
2. **[Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)** – Helpful and harmless human feedback dataset.
3. **[Truthy-DPO-v0.1](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1)** – Preference dataset tailored for Direct Preference Optimization (DPO).
   
> ⚠️ **Note:** The same training script supports both **Truthy-DPO** and **HH-RLHF** out-of-the-box.  
> ⚠️ **AlpacaEval** uses a slightly different format; support is included but **commented out** in the code.

---

## 🧠 Models Evaluated

We tested PROPS on the following pretrained language models:

- **[GPT2-Medium](https://huggingface.co/gpt2-medium)** (355M parameters)
- **[GPT2-Large](https://huggingface.co/gpt2-large)** (774M parameters)
- **[Pythia-1B](https://huggingface.co/EleutherAI/pythia-1b)** – Open LLM from EleutherAI

---

## 🗂️ Repository Structure

PROPS/

├── DP-SGD/ # Code for DP-SGD alignment 

│ ├── Trainer.py # Custom trainer with gradient clipping and noise

│ └── Preference_dataset.py # Dataset wrapper with DP_SGD mechanisms

├── Eval/ # Evaluation utilities 

│ ├── preference_dataset.py         ⚠️(must be fully included)

│ ├── utils.py                      ⚠️(must be fully included)

│ ├── gpt4-eval                     ⚠️(must be fully included)

│ └── Win_Rate.ipynb 


├── Label-DP/ # Randomized Response utilities

├── PROPS/ # Core PROPS training logic

├── SFT_Train/ # Scripts for supervised fine-tuning

│ └── configs/ # YAML config files for training/eval   ⚠️(must be fully included, we provide a simple pythia1B example)

├── README.md # This file

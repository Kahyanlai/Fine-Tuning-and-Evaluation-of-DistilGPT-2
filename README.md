# 🤖 SIT744 Assignment 4 – Fine-Tuning and Evaluation of DistilGPT-2

## 📘 Overview
This project explores how large language models such as **DistilGPT-2** can be adapted to a new writing domain through **fine-tuning** and **parameter-efficient methods**.  

The selected domain was *scientific abstracts in healthcare machine learning*.  
Using only 5 short paragraphs as domain samples, the study evaluates trade-offs between **full fine-tuning**, **modified hyperparameter tuning**, and **Parameter-Efficient Fine-Tuning (LoRA)**.

---

## 🎯 Objectives
1. Evaluate the generative capability of pretrained DistilGPT-2 on a new domain.  
2. Perform **full fine-tuning** and **hyperparameter tuning** on small in-domain data.  
3. Apply **LoRA (Low-Rank Adaptation)** to reduce training cost and avoid overfitting.  
4. Compare models on **perplexity**, fluency, coherence, and training efficiency.

---

## 🧩 Methodology

### 1️⃣ Baseline (P-level Task)
- Model: DistilGPT-2 pretrained checkpoint (`transformers` library).  
- Prompt: “Continue the following passage in the same style…”  
- Generated text shows fluent academic style but limited healthcare context.  
- **Perplexity:** 35.29 on generated text; ~41 average on domain samples.

### 2️⃣ Full Fine-Tuning (C-level Task)
- Fine-tuned DistilGPT-2 on 5 healthcare abstracts.  
- Settings:  
  - Epochs = 3, batch size = 2, LR = 5e-5 → then modified to 1e-4.  
  - Max length = 256 tokens.  
- **Findings:**
  - LR=5e-5: More domain terms but incoherent text.  
  - LR=1e-4: Better coherence and healthcare alignment but overfit behaviour.  
- **Domain perplexity:** Improved from 26.2 → 18.1, but **generated perplexity worsened** (>110).  

### 3️⃣ Parameter-Efficient Fine-Tuning (D-level Task)
- Method: **LoRA (Low-Rank Adaptation)** — adds trainable rank-8 adapter matrices in attention layers while freezing 99.8% of weights.  
- Trainable parameters reduced from **82M → 0.16M (0.2%)**.  
- Config: `r=8, α=32, bias="all", lr=5e-4`.  
- **Result:**  
  - LoRA training time: 1.14s  
  - Average domain perplexity: 38.5  
  - Generated text remained fluent, structured, and academic in tone.

---

## 📊 Quantitative Comparison

| Method | Trainable Params | Train Time (s) | Inference (s) | Gen. PPL ↓ | Domain PPL ↓ | Qualitative Summary |
|--------|------------------|----------------|----------------|-------------|---------------|----------------------|
| Baseline (Pretrained) | 0 | 0 | 1.24 | **35.29** | 41.08 | Fluent but generic |
| Full Fine-Tuning (5e-5) | 82M | 1.55 | 0.94 | 111.03 | 26.22 | Domain-ish, incoherent |
| Modified Fine-Tuning (1e-4) | 82M | 3.38 | 1.16 | 111.52 | **18.07** | Domain-aligned, overfit |
| **LoRA (r=8, α=32)** | **0.16M (0.2%)** | **1.14** | **0.92** | **110.06** | 38.49 | Fluent, balanced |

---

## ⚖️ Insights
- **Full fine-tuning** yields strong domain alignment but reduced fluency and high overfitting on small data.  
- **LoRA** preserves general fluency while adapting modestly to the new domain, with minimal computational cost.  
- **Trade-off:** fine-tuning = specificity + overfit; LoRA = generality + fluency.  

🧩 *With larger datasets, LoRA would likely achieve domain-specific adaptation with less overfitting than full fine-tuning.*

---

## 🛠️ Tech Stack
- **Language:** Python 3.9  
- **Libraries:** PyTorch • Hugging Face Transformers • Datasets • PEFT  
- **Hardware:** Google Colab GPU  
- **Metrics:** Perplexity, qualitative style comparison, training/inference time  

---

## 🚀 How to Run
```bash
pip install torch transformers datasets peft
python 223330692_sit744_assignment4_solution.py

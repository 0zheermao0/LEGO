# LIGHT: LLM-guided Graph Expert Routing for Semi-supervised Domain Generalization

---

This repository contains the official implementation of the paper: **LIGHT: LLM-guided Graph Expert Routing for Semi-supervised Domain Generalization** (NeurIPS 2025)[cite: 1].

## 📖 Abstract

Graph Neural Networks (GNNs) excel in graph machine learning but face challenges with distribution shifts and limited labeled data[cite: 1]. To address this, we explore **semi-supervised domain generalization**, aiming to boost GNN performance on unseen graphs using both labeled and unlabeled data[cite: 2]. We introduce **LIGHT (LLM-Guided Graph Expert Routing)**, a novel approach that uses Large Language Models (LLMs) as judges to guide a multi-hop graph Mixture-of-Experts (MoE) framework[cite: 3, 4]. LIGHT employs diverse graph experts exploring various neighborhood depths[cite: 5]. LLMs provide context-aware routing guidance by identifying the most reliable experts for critical nodes, enhancing generalizability through knowledge distillation[cite: 6]. We also introduce an expert-aware dynamic pseudo-labeling strategy to tackle label scarcity[cite: 7]. Experiments show LIGHT's effectiveness compared to existing methods[cite: 8].

---

## 🖼️ Framework Overview

LIGHT leverages LLMs to guide a Mixture-of-Experts (MoE) framework for robust semi-supervised domain generalization on graphs.

%% ![Framework Overview](https://storage.googleapis.com/gemini-generative-ai-prod-us-west1-0000/images/2024-05-26/1601053b-e1c7-436d-8a2a-db7d47f9e422.png) %%

* **Input**: An input graph with labeled and unlabeled nodes[cite: 47].
* **Multi-hop MoE**: Node contexts are decoupled into different hops (0-hop, 1-hop, 2-hop) and fed into an MoE framework[cite: 49]. Each expert specializes in a specific neighborhood depth.
* **LLM-as-Judge**: Node contexts and expert predictions are transformed into prompts for an LLM[cite: 50]. The LLM acts as a judge, selecting the most reliable expert for specific nodes.
* **Knowledge Distillation**: The LLM's judgments guide a lightweight routing function via knowledge distillation, focusing on influential 'anchor' nodes[cite: 51, 95].
* **Expert-aware Pseudo-labeling**: A dynamic pseudo-labeling strategy selects high-confidence unlabeled nodes for additional training, guided by expert outputs[cite: 52, 101].

---

## ✨ Key Features

* **LLM-Guided Routing**: Utilizes the zero-shot reasoning of LLMs to dynamically select the most suitable GNN expert for each node, improving adaptability[cite: 6, 79].
* **Multi-Hop Mixture-of-Experts (MoE)**: Employs diverse GNN experts, each focusing on different neighborhood depths (hops), to capture varied structural information and handle distribution shifts[cite: 5, 70].
* **Knowledge Distillation**: Transfers the LLM's complex routing decisions to a lightweight, efficient student model using an influence-maximization-inspired strategy[cite: 92, 95].
* **Dynamic Pseudo-Labeling**: Addresses label scarcity by generating high-quality pseudo-labels using an expert-aware dynamic threshold, enhancing semi-supervised learning[cite: 7, 101, 102].
* **Diversity Promotion**: Includes an objective to encourage diversity among experts, ensuring they capture distinct aspects of graph structure[cite: 104].

---

## 🔧 Installation

1.  **Clone the repository:**

2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n light_env python=3.9
    conda activate light_env
    ```

3.  **Install PyTorch and PyTorch Geometric:**
    Follow the official instructions based on your CUDA version:
    * [PyTorch](https://pytorch.org/get-started/locally/)
    * [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

4.  **Install other dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create a `requirements.txt` file based on the imports in the Python files, including `numpy`, `scipy`, `scikit-learn`, `wandb`, `ollama`, `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`.)*

5.  **Set up Ollama (for LLM inference):**
    * Install and run Ollama: [https://ollama.com/](https://ollama.com/)
    * Pull the desired model (e.g., `qwen2.5:7b` as used in `main.py`):
        ```bash
        ollama pull qwen2.5:7b
        ```
    * (Optional) If running Ollama on a different host, update `os.environ['OLLAMA_HOST']` in `main.py`.

---

## 📊 Datasets

We use benchmark datasets from ArnetMiner[cite: 106]:
* **ACMv9**
* **Citationv1**
* **DBLPv7**

These datasets involve classifying papers into five research areas[cite: 107].

**Preparation:**
1.  Download the datasets (or provide instructions/links).
2.  Place the `.mat` files (as suggested by `DomainData.py`) into the `data/<dataset_name>/` directory (e.g., `data/dblpv7/dblpv7.mat`).
3.  The code will process these and save `data.pt` files in `data/<dataset_name>/processed/`.
4.  The code uses a `tmp/` directory to cache precomputed PPMI matrices. Ensure this directory exists: `mkdir tmp`.

---

## 🚀 How to Run

Use the `main.py` script to train and evaluate the LIGHT model.

**Example Training Command:**

```bash
python main.py \
    --source dblpv7 \
    --target citationv1 \
    --seed 1 \
    --learning_rate 1e-3 \
    --weight_decay 2e-3 \
    --encoder_dim 512 \
    --expert_num 3 \
    --llm qwen2.5:7b \
    --uncertainty_k 100 \
    --llm_interval 10 \
    --wandb_project "LIGHT-NeurIPS25" \
    --wandb_entity "your_wandb_username"

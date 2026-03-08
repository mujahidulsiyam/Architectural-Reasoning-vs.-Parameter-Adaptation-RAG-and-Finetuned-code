# Architectural-Reasoning-vs.-Parameter-Adaptation-RAG-and-Finetuned-code
This repository contains the official implementation for the paper: "Architectural Reasoning vs. Parameter Adaptation: A Comprehensive Study of SLM Performance in RAG Pipelines at Thick Edge."

1. Overview:
   This study compares Retrieval-Augmented Generation (RAG) and Instruction Fine-Tuning (FT) across three model families at 1B, 3B, and 7B scales. Key Finding: RAG       improves generative fluency and empathy ($p < 0.001$), while Fine-Tuning enhances diagnostic classification stability.
3.  PrerequisitesHardwareGPU:
    Dual NVIDIA Tesla T4 (2 x 15GB VRAM) or equivalent.Memory: Minimum 16GB RAM for handling 7B models in 4-bit quantization.
4.  Software & Environment:
    Install the required libraries to match our experimental environment:

    pip install -r requirement.txt

    (This includes transformers, peft, bitsandbytes, faiss-cpu, and sentence-transformers)
5.  Dataset Setup:
    Download the Mental Health Counseling Conversations dataset from Kaggle .Ensure the file is named train_data.csv and placed in the /input/                            directory.The script will automatically perform a 5-fold cross-validation split using a fixed random_seed=42.
    
7.  How to Reproduce the 18 Experiments:
    Instead of 18 separate files, we provide two Universal Scripts that use a "switch" to change between model variants.
    Step 1: Choose a Paradigm
    Use universal_finetuned.py for Instruction Fine-Tuning experiments.
    Use Universal_RAG.py for Retrieval-Augmented Generation experiments.
    
    Step 2: Select a Model Variant
    Open the script and modify the SELECTED_MODEL variable at the top:
    # Options: "llama_1b", "llama_3b", "llama_7b", "falcon_1b", etc.
    SELECTED_MODEL = "qwen_7b"

    Step 3: Run the ScriptExecute the script in your environment (e.g., Kaggle or local terminal).
    For Fine-Tuning: The script applies QLoRA ($r=16, \alpha=32$) and runs 3 epochs per fold.
    For RAG: The script builds a FAISS index using all-MiniLM-L6-v2 and retrieves top-k=3 context units.
    
9.  Evaluation & Results:
    The scripts automatically calculate the following metrics for each fold:
    Lexical: BLEU and ROUGE-L.
    Semantic: BERTScore F1 (using roberta-large).
    Stability: Mean ($\mu$) and Standard Deviation ($\sigma$) across all 5 folds.

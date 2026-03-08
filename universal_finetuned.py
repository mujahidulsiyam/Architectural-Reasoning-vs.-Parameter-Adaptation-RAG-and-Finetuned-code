# UNIVERSAL CONFIGURATION
# Change SELECTED_MODEL to any of these 9 options to reproduce specific results
MODELS = {
    "llama_1b": "meta-llama/Llama-3.2-1B",
    "llama_3b": "meta-llama/Llama-3.2-3B",
    "llama_7b": "meta-llama/Llama-3.1-8B", 
    "falcon_1b": "tiiuae/falcon-1b",
    "falcon_3b": "tiiuae/falcon-3b",
    "falcon_7b": "tiiuae/falcon-7b",
    "qwen_1b": "Qwen/Qwen2.5-1.5B",
    "qwen_3b": "Qwen/Qwen2.5-3B",
    "qwen_7b": "Qwen/Qwen2.5-7B"
}

SELECTED_MODEL = "qwen_7b"  # <--- CHANGE THIS FOR EACH RUN
model_id = MODELS[SELECTED_MODEL]


# Dependencies & Imports

!pip install -q accelerate peft bitsandbytes transformers trl evaluate rouge-score bert-score
import os, gc, torch, pandas as pd, numpy as np, evaluate
from datasets import Dataset
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig
from trl import SFTTrainer
from time import perf_counter


# Data Loading & Preprocessing [cite: 107, 115]

df = pd.read_csv("/kaggle/input/5000-data/train_data.csv").dropna()
df["text"] = df.apply(lambda x: f"<|im_start|>user\n{x['Context']} <|im_end|>\n<|im_start|>assistant\n{x['Response']}<|im_end|>", axis=1)


# 5-Fold CV Loop
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
    print(f"\n--- Training Fold {fold} ---")
    train_ds = Dataset.from_pandas(df.iloc[train_idx][["text"]])
    
    # Model Loading with 4-bit Quantization 
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # QLoRA Config 
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    # Training Arguments 
    args = TrainingArguments(output_dir=f"ft_{SELECTED_MODEL}_fold_{fold}", per_device_train_batch_size=1,
                             gradient_accumulation_steps=4, learning_rate=2e-4, num_train_epochs=3, fp16=True)

    trainer = SFTTrainer(model=model, train_dataset=train_ds, peft_config=peft_config, args=args,
                         data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    
    trainer.train()
    
    # Save Metrics & Cleanup
    fold_results.append({"fold": fold, "model": SELECTED_MODEL, "loss": trainer.state.log_history[-1].get("train_loss")})
    del model, trainer; torch.cuda.empty_cache(); gc.collect()

pd.DataFrame(fold_results).to_csv(f"results_ft_{SELECTED_MODEL}.csv", index=False)

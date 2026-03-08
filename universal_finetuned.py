# UNIVERSAL CONFIGURATION
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

SELECTED_MODEL = "llama_3b"  # <--- CHANGE THIS FOR EACH RUN
model_id = MODELS[SELECTED_MODEL]

# 1. Install Dependencies
!pip install -q accelerate peft bitsandbytes transformers trl evaluate rouge-score bert-score
!pip install -q git+https://github.com/huggingface/trl.git

# 2. Imports & Setup
import os, gc, torch, warnings, pandas as pd, numpy as np
from datasets import Dataset
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from time import perf_counter
import evaluate

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Login
SECRET_LABEL = "HF_TOKEN"
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret(SECRET_LABEL)
login(token=hf_token)


# 3. Helper Functions
def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 256

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer

def compute_accuracy_vectorized(preds, refs, tokenizer):
    pred_tokens = tokenizer(preds, padding=True, truncation=True, return_tensors="pt").input_ids
    ref_tokens = tokenizer(refs, padding=True, truncation=True, return_tensors="pt").input_ids
    min_len = min(pred_tokens.shape[1], ref_tokens.shape[1])
    correct = (pred_tokens[:, :min_len] == ref_tokens[:, :min_len]).sum().item()
    total = pred_tokens[:, :min_len].numel()
    return correct / total if total > 0 else 0.0

def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir): return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints: return None
    latest = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return os.path.join(output_dir, latest)

# 4. Load Data 

CSV_PATH = "/kaggle/input/5000-data/train_data.csv"
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"Context": "prompt", "Response": "completion"})
df = df.dropna(subset=["prompt", "completion"])

# Formatting 
df["text"] = df.apply(
    lambda x: f"<|im_start|>user\n{x['prompt']} <|im_end|>\n<|im_start|>assistant\n{x['completion']}<|im_end|>",
    axis=1
)

# 5. K-Fold CV Loop 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

# Load metrics 
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
    print(f"\n===== Fold {fold}/5 =====")
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    train_dataset = Dataset.from_pandas(train_df[["text"]])

    # Cleanup & Load 
    gc.collect(); torch.cuda.empty_cache()
    model, tokenizer = get_model_and_tokenizer(model_id)
    output_dir = f"/kaggle/working/ft_{SELECTED_MODEL}_fold_{fold}"

    # QLoRA Config 
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Training Arguments 
    args = TrainingArguments(
        output_dir=output_dir, per_device_train_batch_size=1, gradient_accumulation_steps=4,
        optim="paged_adamw_32bit", learning_rate=2e-4, lr_scheduler_type="cosine",
        num_train_epochs=3, fp16=True, save_strategy="steps", save_steps=100, save_total_limit=2,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model, train_dataset=train_dataset, peft_config=peft_config, args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # Train
    resume = find_latest_checkpoint(output_dir)
    start = perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume)
    duration = round(perf_counter() - start, 2)

    # 6. Detailed Evaluation 
    print(f"Evaluating Fold {fold}...")
    model.eval()
    preds, refs = [], val_df["completion"].tolist()
    prompts = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n" for p in val_df["prompt"].tolist()]

    for i in range(0, len(prompts), 4):
        batch = prompts[i:i+4]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
        
        for out in outputs:
            text = tokenizer.decode(out, skip_special_tokens=True)
            text = text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
            preds.append(text)

    # Compute Metrics 
    b = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])["bleu"]
    r = rouge_metric.compute(predictions=preds, references=refs)["rougeL"]
    bs = bertscore_metric.compute(predictions=preds, references=refs, lang="en")
    acc = compute_accuracy_vectorized(preds, refs, tokenizer)
    
    avg_bs = sum(bs["f1"]) / len(bs["f1"])
    print(f"Fold {fold} -> BLEU: {b:.4f}, ROUGE-L: {r:.4f}, BERT-F1: {avg_bs:.4f}, Acc: {acc:.4f}")

    fold_results.append({
        "fold": fold, "model": SELECTED_MODEL, "loss": train_result.training_loss,
        "bleu": b, "rougeL": r, "bertscore_f1": avg_bs, "accuracy": acc, "time": duration
    })

    # Save & Cleanup
    trainer.model.save_pretrained(f"{output_dir}/final")
    del model, trainer; torch.cuda.empty_cache(); gc.collect()

# 7. Final Summary
summary_df = pd.DataFrame(fold_results)
summary_df.to_csv(f"/kaggle/working/results_{SELECTED_MODEL}.csv", index=False)
print(summary_df)

# UNIVERSAL RAG CONFIGURATION
MODELS = {
    "llama_1b": "meta-llama/Llama-3.2-1B",
    "llama_3b": "meta-llama/Llama-3.2-3B",
    "llama_7b": "meta-llama/Llama-2-7b-chat-hf",
    "falcon_1b": "tiiuae/falcon-1b",
    "falcon_3b": "tiiuae/falcon-3b",
    "falcon_7b": "tiiuae/falcon-7b",
    "qwen_1b": "Qwen/Qwen2.5-1.5B",
    "qwen_3b": "Qwen/Qwen2.5-3B",
    "qwen_7b": "Qwen/Qwen2.5-7B"
}

SELECTED_MODEL = "llama_7b" # <--- CHANGE THIS FOR EACH RUN
model_id = MODELS[SELECTED_MODEL]

# 1. Install Dependencies [cite: 123, 124]
!pip install -q faiss-cpu sentence-transformers transformers accelerate bitsandbytes evaluate bert-score


# 2. Imports & Setup
import os, gc, torch, pandas as pd, numpy as np, faiss, evaluate, json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.model_selection import KFold
from time import perf_counter
from huggingface_hub import login

# Login (Use your Kaggle Secret)
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
login(token=user_secrets.get_secret("HF_TOKEN"))


# 3. Load Resources 

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") # [cite: 123]
df = pd.read_csv("/kaggle/input/5000-data/train_data.csv").rename(columns={"Context": "prompt", "Response": "completion"}).dropna()

# Load Metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

# Load Model (4-bit NF4) 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


# 4. RAG Generation Logic 

def rag_generate(question, index, train_df, top_k=3):
    q_vec = embedder.encode([question])
    D, I = index.search(np.array(q_vec).astype('float32'), top_k) # [cite: 124, 142]
    
    retrieved = [f"Q: {train_df.iloc[idx]['prompt']}\nA: {train_df.iloc[idx]['completion']}" for idx in I[0]]
    context = "\n\n".join(retrieved)
    
    # Standardized Prompt Template
    prompt = f"<s>[INST] Use these examples for style and context:\n{context}\n\nAnswer this: {question} [/INST]\nA:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True) # [cite: 158]
    
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("[/INST]")[-1].replace("A:", "").strip()


# 5. 5-Fold Evaluation Loop 

kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
    print(f"\n--- RAG Fold {fold}/5 ---")
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Build FAISS Index 
    train_embeddings = embedder.encode(train_df["prompt"].tolist(), show_progress_bar=True)
    index = faiss.IndexFlatL2(train_embeddings.shape[1]) # [cite: 142, 157]
    index.add(np.array(train_embeddings).astype('float32'))

    preds, refs = [], val_df["completion"].tolist()
    start_time = perf_counter()

    # Inference on Validation Set
    for q in val_df["prompt"].tolist():
        preds.append(rag_generate(q, index, train_df))

    total_time = perf_counter() - start_time

    # 6. Evaluation Metrics 
    b = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])["bleu"] # [cite: 197]
    r = rouge_metric.compute(predictions=preds, references=refs)["rougeL"] # [cite: 198]
    bs = bertscore_metric.compute(predictions=preds, references=refs, lang="en") # [cite: 200, 201]
    
    avg_bs = np.mean(bs["f1"])
    print(f"Fold {fold} -> BLEU: {b:.4f}, ROUGE-L: {r:.4f}, BERT-F1: {avg_bs:.4f}")

    all_results.append({
        "fold": fold, "model": SELECTED_MODEL, "bleu": b, 
        "rougeL": r, "bertscore_f1": avg_bs, "time": total_time
    })

    # Cleanup
    del index; gc.collect(); torch.cuda.empty_cache()

# 7. Summary

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"/kaggle/working/results_rag_{SELECTED_MODEL}.csv", index=False)
print(results_df)

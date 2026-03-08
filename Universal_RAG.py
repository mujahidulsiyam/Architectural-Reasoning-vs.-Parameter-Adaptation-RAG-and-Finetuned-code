# ==============================
# UNIVERSAL RAG CONFIGURATION
# ==============================
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

# ==============================
# Dependencies & Imports
# ==============================
!pip install -q faiss-cpu sentence-transformers transformers accelerate bitsandbytes evaluate bert-score
import os, gc, torch, pandas as pd, numpy as np, faiss, evaluate
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.model_selection import KFold

# ==============================
# RAG Setup [cite: 123, 139, 142]
# ==============================
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
df = pd.read_csv("/kaggle/input/5000-data/train_data.csv").dropna()

# Load 4-bit Model [cite: 156]
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# ==============================
# Evaluation Loop [cite: 88, 332]
# ==============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_rag_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Build FAISS Index [cite: 124, 142]
    train_embeddings = embedder.encode(train_df["Context"].tolist())
    index = faiss.IndexFlatL2(train_embeddings.shape[1])
    index.add(np.array(train_embeddings).astype('float32'))

    preds = []
    # Simple RAG Inference [cite: 143, 144]
    for question in val_df["Context"].tolist()[:10]: # Testing first 10 for speed
        D, I = index.search(embedder.encode([question]), k=3) # top-k=3 [cite: 152, 339]
        context = "\n".join([train_df.iloc[idx]["Response"] for idx in I[0]])
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=128)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))

    all_rag_results.append({"fold": fold, "model": SELECTED_MODEL, "samples": len(preds)})
    del index; torch.cuda.empty_cache(); gc.collect()

pd.DataFrame(all_rag_results).to_csv(f"results_rag_{SELECTED_MODEL}.csv", index=False)

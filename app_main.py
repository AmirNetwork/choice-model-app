# File: app_neo3.py
#!/usr/bin/env python3
"""
Flask app using RAG with LoRA-finetuned GPT-Neo for chat/code assistance.
"""
import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime

import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, render_template, session
from werkzeug.utils import secure_filename

# --- Config ---
BASE_DIR = Path(__file__).parent
STORAGE = BASE_DIR / 'storage'
UPLOAD_DIR = STORAGE / 'uploads'
LOG_FILE = STORAGE / 'chat.log'
REF_DIR = STORAGE / 'references'
LORA_DIR = BASE_DIR / 'lora_finetuned'
CACHE_DIR = BASE_DIR / 'hf_cache'
ALLOWED = {'csv','xlsx'}
TOP_K = 5
MAX_IN = 1792
MAX_OUT = 256

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)
app.secret_key = os.urandom(16)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)

# --- Build RAG index on startup ---
embedder = SentenceTransformer("microsoft/codebert-base")
chunks, sources = [], []
for f in sorted(os.listdir(REF_DIR)):
    if f.endswith('.json'):
        r = json.loads(open(REF_DIR/f, encoding='utf-8').read())
        text = r.get('paper_text','')
        code = r.get('code_snippet','')
        segs = (text+'\n\n'+code).split('\n\n')
        for s in segs:
            if len(s.strip())>=100:
                chunks.append(s.strip())
                sources.append(r.get('title',f))
vecs = embedder.encode(chunks, convert_to_numpy=True)
idx = faiss.IndexFlatL2(vecs.shape[1])
idx.add(vecs)
logging.info(f"Built FAISS index with {len(chunks)} chunks.")

# --- Load model/tokenizer ---
tok = AutoTokenizer.from_pretrained(str(LORA_DIR), cache_dir=str(CACHE_DIR))
model = AutoModelForCausalLM.from_pretrained(str(LORA_DIR), cache_dir=str(CACHE_DIR))
model.eval()


def retrieve_ctx(query: str):
    qv = embedder.encode([query], convert_to_numpy=True)
    _, ids = idx.search(qv, TOP_K)
    ctxs, title = [], sources[ids[0][0]]
    for i in ids[0]:
        ctxs.append(f"# Source: {sources[i]}\n{chunks[i]}")
    return title, '\n\n'.join(ctxs)


def generate(user_msg: str) -> str:
    t0 = time.time()
    title, context = retrieve_ctx(user_msg)
    logging.info(f"Retrieval ({title}) in {time.time()-t0:.2f}s")
    prompt = (
        f"You are an expert assistant.\nTask: {title}\n"
        f"Context:\n{context}\n"
        f"Instruction:\n{user_msg}\n"
        "Response:\n"
    )
    inp = tok(prompt, return_tensors='pt', truncation=True, max_length=MAX_IN)
    t1 = time.time()
    out = model.generate(
        **inp,
        max_new_tokens=MAX_OUT,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        pad_token_id=tok.eos_token_id
    )
    resp = tok.decode(out[0], skip_special_tokens=True)[len(prompt):]
    logging.info(f"Generation in {time.time()-t1:.2f}s")
    return resp


def log_chat(user, q, a):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now()} | {user}: {q} -> {a}\n")

@app.route('/', methods=['GET','POST'])
def chat():
    reply = None
    if request.method=='POST':
        msg = request.form['message']
        reply = generate(msg)
        log_chat(session.get('user','anon'), msg, reply)
    return render_template('chat.html', response=reply)

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if f and '.' in f.filename and f.filename.rsplit('.',1)[1] in ALLOWED:
        fn = secure_filename(f.filename)
        p = UPLOAD_DIR/ fn
        f.save(p)
        df = pd.read_csv(p) if fn.endswith('.csv') else pd.read_excel(p)
        session['preview'] = df.head().to_json()
        return "Uploaded"
    return "Invalid"

if __name__=='__main__':
    UPLOAD_DIR.mkdir(exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
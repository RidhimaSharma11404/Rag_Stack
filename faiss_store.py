"""
faiss_store.py — compatibility shim
vector_db.py now runs BM25+TF-IDF fully offline.
This file keeps load_faiss/search_faiss so nothing breaks
if FAISS is installed in your packages folder.
"""
try:
    import faiss, numpy as np, pickle, os
    from sentence_transformers import SentenceTransformer

    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    INDEX_PATH = os.path.join(BASE_DIR, "faiss_db", "faiss.index")
    META_PATH  = os.path.join(BASE_DIR, "faiss_db", "faiss_meta.pkl")

    _index = _answers = _questions = None

    def load_faiss(model, questions, answers):
        global _index, _answers, _questions
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            _index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f: m = pickle.load(f)
            _questions, _answers = m["questions"], m["answers"]
            if _index.ntotal == len(_questions):
                print(f"[FAISS] Loaded {_index.ntotal:,} records.")
                return _index
        emb = model.encode(questions, show_progress_bar=True, batch_size=64).astype("float32")
        faiss.normalize_L2(emb)
        _index = faiss.IndexFlatIP(emb.shape[1]); _index.add(emb)
        faiss.write_index(_index, INDEX_PATH)
        with open(META_PATH,"wb") as f: pickle.dump({"questions":questions,"answers":answers},f)
        _questions, _answers = questions, answers
        print(f"[FAISS] Built {len(questions):,} records.")
        return _index

    def search_faiss(model, query, top_k=5):
        emb = model.encode([query]).astype("float32"); faiss.normalize_L2(emb)
        D, I = _index.search(emb, min(top_k, _index.ntotal))
        return {"answer": _answers[int(I[0][0])], "question": _questions[int(I[0][0])],
                "score": round(float(D[0][0]), 4)}

except ImportError:
    def load_faiss(model=None, questions=None, answers=None):
        print("[FAISS] Not installed — BM25+TF-IDF active."); return None
    def search_faiss(model=None, query=None, top_k=5):
        return {"answer": None, "score": 0.0, "question": ""}
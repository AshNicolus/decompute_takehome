import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

clf_results = pd.read_csv("clf_results.csv")

def load_kb(kb_dir="kb"):
    chunks = []
    for fname in sorted(os.listdir(kb_dir)):
        if not fname.endswith(".txt"):
            continue
        source = fname.replace(".txt", "")
        fpath = os.path.join(kb_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            raw = f.read().strip()

        sentences = re.split(r"(?<=[.!?])\s+", raw)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 5]

        for idx, sent in enumerate(sentences):
            chunks.append({
                "source": source,
                "chunk_id": f"{source}_s{idx}",
                "text": sent,
            })
    return chunks

kb_chunks = load_kb("kb")
print(f"[KB] Loaded {len(kb_chunks)} chunks from {len(set(c['source'] for c in kb_chunks))} files")
for src, cnt in sorted({c["source"]: 0 for c in kb_chunks}.items()):
    n = sum(1 for c in kb_chunks if c["source"] == src)
    print(f"  {src:<15} {n} chunks")

chunk_texts = [c["text"] for c in kb_chunks]

retriever = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=8000,
    stop_words="english",
    sublinear_tf=True,
)
chunk_matrix = retriever.fit_transform(chunk_texts)

def retrieve(query, top_k=3, min_score=0.05):
    q_vec = retriever.transform([query])
    scores = cosine_similarity(q_vec, chunk_matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top_idx:
        results.append({
            "score": float(scores[idx]),
            "source": kb_chunks[idx]["source"],
            "chunk_id": kb_chunks[idx]["chunk_id"],
            "text": kb_chunks[idx]["text"],
        })
    return results

CATEGORY_HINTS = {
    "billing": "billing subscription invoice payment refund",
    "technical_issue": "technical error crash bug troubleshoot API status",
    "account_access": "account password login reset two-factor authentication",
    "feature_request": "feature dashboard analytics workflow productivity",
    "general_query": "platform onboarding getting started service terms",
}

ABSTAIN_THRESHOLD = 0.05

def generate_response(ticket_text, predicted_category, retrieved_chunks):
    if not retrieved_chunks or retrieved_chunks[0]["score"] < ABSTAIN_THRESHOLD:
        return None

    sources_cited = list(dict.fromkeys(c["source"] for c in retrieved_chunks))
    kb_content = " ".join(c["text"] for c in retrieved_chunks[:2])

    if len(kb_content) > 350:
        kb_content = kb_content[:350].rsplit(" ", 1)[0] + "..."

    category_label = predicted_category.replace("_", " ").title()

    response = (
        f"Thank you for reaching out. Based on your {category_label} inquiry, "
        f"here is information from our knowledge base that should help:\n\n"
        f"{kb_content}\n\n"
        f"If this does not fully resolve your issue, please contact our support team "
        f"directly for further assistance.\n\n"
        f"[Sources: {', '.join(sources_cited)}]"
    )
    return response

print("\n[Running retrieval + generation pipeline on eval tickets...]")

records = []
for _, row in clf_results.iterrows():
    ticket_id = row["ticket_id"]
    text = row["text"]
    category = row["predicted_category"]
    priority = row["predicted_priority"]
    cat_conf = row["cat_confidence"]
    pri_conf = row["pri_confidence"]

    hint = CATEGORY_HINTS.get(category, "")
    query = f"{text} {hint}"

    retrieved = retrieve(query, top_k=3)
    top_score = retrieved[0]["score"] if retrieved else 0.0

    classifier_conf = float(cat_conf)
    should_abstain = (top_score < ABSTAIN_THRESHOLD) or (classifier_conf < 0.30)

    draft = generate_response(text, category, retrieved) if not should_abstain else None

    retrieval_conf_norm = min(top_score / 0.4, 1.0)
    combined_conf = (
        2 * classifier_conf * retrieval_conf_norm
        / (classifier_conf + retrieval_conf_norm + 1e-9)
    ) if not should_abstain else 0.0

    records.append({
        "ticket_id": ticket_id,
        "predicted_category": category,
        "predicted_priority": priority,
        "top_3_retrieved_sources": "|".join(list(dict.fromkeys(r["source"] for r in retrieved))),
        "top_3_chunk_ids": "|".join(r["chunk_id"] for r in retrieved),
        "top_retrieval_score": round(top_score, 4),
        "draft_response": draft if draft else "ABSTAIN: Insufficient information to provide a reliable answer.",
        "confidence_score": round(combined_conf, 4),
        "abstain_flag": int(should_abstain),
        "cat_confidence": round(cat_conf, 4),
        "pri_confidence": round(pri_conf, 4),
    })

rag_results = pd.DataFrame(records)
rag_results.to_csv("rag_results.csv", index=False)

print(f"\n[Pipeline complete]")
print(f"  Total tickets     : {len(rag_results)}")
print(f"  Abstained         : {rag_results['abstain_flag'].sum()}")
print(f"  Answered          : {(~rag_results['abstain_flag'].astype(bool)).sum()}")
print(f"  Avg confidence    : {rag_results['confidence_score'].mean():.3f}")

print("\n[Sample outputs]")
for _, row in rag_results.head(3).iterrows():
    print(f"\n  Ticket {row['ticket_id']}: [{row['predicted_category']}] [{row['predicted_priority']}]")
    print(f"  Sources : {row['top_3_retrieved_sources']}")
    print(f"  Conf    : {row['confidence_score']:.3f}  Abstain: {bool(row['abstain_flag'])}")
    draft_preview = str(row['draft_response'])[:120] + "..."
    print(f"  Draft   : {draft_preview}")

print("\n[Saved rag_results.csv]\n")
print("[Part 3 complete]\n")
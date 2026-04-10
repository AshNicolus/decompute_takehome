import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, brier_score_loss
)
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("tickets_train.csv")
rag_res = pd.read_csv("rag_results.csv")

train["text"] = (train["subject"].fillna("") + " " + train["message"].fillna("")).str.strip()
X_train = train["text"].tolist()
y_cat = train["category"].tolist()
y_pri = train["priority"].tolist()

print("=" * 60)
print("PART 4: EVALUATION")
print("=" * 60)

URGENCY_WORDS = {
    "crash","crashing","broken","error","critical","urgent","cannot","unable",
    "failed","failure","down","outage","immediately","asap","emergency","lost",
    "data","security","breach","hack","charge","overcharge","wrong","incorrect",
}
LOW_PRI_WORDS = {
    "feature","request","suggestion","consider","would","could","nice","improve",
    "idea","wish","wondering","question","how","general","info","learn",
}

class UrgencyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        rows = []
        for text in X:
            words = set(text.lower().split())
            rows.append([
                len(words & URGENCY_WORDS),
                len(words & LOW_PRI_WORDS),
                int("!" in text),
                len(text.split()),
            ])
        return np.array(rows, dtype=float)

def baseline_pipeline(cw=None):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,1), max_features=3000,
                                   stop_words="english", sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=1000, class_weight=cw, C=1.0)),
    ])

def improved_pipeline(cw=None):
    feat = FeatureUnion([
        ("word", Pipeline([("t", TfidfVectorizer(ngram_range=(1,2), max_features=5000,
                                                  stop_words="english", sublinear_tf=True))])),
        ("char", Pipeline([("t", TfidfVectorizer(ngram_range=(3,5), max_features=3000,
                                                  sublinear_tf=True, analyzer="char_wb"))])),
        ("urg",  UrgencyFeatures()),
    ])
    svc = LinearSVC(max_iter=2000, class_weight=cw, C=0.5)
    cal = CalibratedClassifierCV(svc, cv=5, method="sigmoid")
    return Pipeline([("features", feat), ("clf", cal)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n[A. Classification Performance — 5-fold CV on training set]")
print(f"\n{'Model':<30} {'Cat F1-macro':>14} {'Pri F1-macro':>14}")
print("-" * 60)

for name, cat_pipe, pri_pipe in [
    ("Baseline (TF-IDF + LogReg)",   baseline_pipeline(),   baseline_pipeline("balanced")),
    ("Improved (TF-IDF + SVM + cal)", improved_pipeline(), improved_pipeline("balanced")),
]:
    cat_cv = cross_val_predict(cat_pipe, X_train, y_cat, cv=cv)
    pri_cv = cross_val_predict(pri_pipe, X_train, y_pri, cv=cv)
    cat_f1 = f1_score(y_cat, cat_cv, average="macro")
    pri_f1 = f1_score(y_pri, pri_cv, average="macro")
    cat_ac = accuracy_score(y_cat, cat_cv)
    pri_ac = accuracy_score(y_pri, pri_cv)
    print(f"{name:<30} {cat_f1:>10.3f} ({cat_ac:.2%}) {pri_f1:>10.3f} ({pri_ac:.2%})")

print("\n[Improved model — Category report]")
imp_cat = improved_pipeline()
imp_pri = improved_pipeline("balanced")
cat_cv  = cross_val_predict(imp_cat, X_train, y_cat, cv=cv)
pri_cv  = cross_val_predict(imp_pri, X_train, y_pri, cv=cv)
print(classification_report(y_cat, cat_cv, digits=3))

print("[Improved model — Priority report]")
print(classification_report(y_pri, pri_cv, digits=3))

print("[Confusion matrix — Category]")
cats = sorted(set(y_cat))
cm = confusion_matrix(y_cat, cat_cv, labels=cats)
short = [c[:10] for c in cats]
print(f"{'':22}" + "".join(f"{s:>12}" for s in short))
for i, c in enumerate(cats):
    print(f"{c:<22}" + "".join(f"{v:>12}" for v in cm[i]))

print("\n" + "=" * 60)
print("[B. Retrieval Quality Analysis]")
print("=" * 60)

scores = rag_res["top_retrieval_score"].dropna()
print(f"\n  Retrieval score stats:")
print(f"    Min    : {scores.min():.4f}")
print(f"    Mean   : {scores.mean():.4f}")
print(f"    Median : {scores.median():.4f}")
print(f"    Max    : {scores.max():.4f}")
print(f"    Std    : {scores.std():.4f}")

buckets = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]
print("\n  Score distribution:")
for lo, hi in buckets:
    cnt = ((scores >= lo) & (scores < hi)).sum()
    pct = 100 * cnt / len(scores)
    print(f"    [{lo:.2f}, {hi:.2f})  {cnt:>3} tickets ({pct:.0f}%)")

all_sources = []
for s in rag_res["top_3_retrieved_sources"].dropna():
    all_sources.extend(s.split("|"))
from collections import Counter
src_counts = Counter(all_sources)
print("\n  Top retrieved sources (across all top-3):")
for src, cnt in src_counts.most_common():
    print(f"    {src:<20} {cnt:>3} retrievals")

print("\n  Category → most-retrieved source (alignment check):")
for cat in sorted(rag_res["predicted_category"].unique()):
    subset = rag_res[rag_res["predicted_category"] == cat]
    srcs = []
    for s in subset["top_3_retrieved_sources"].dropna():
        srcs.append(s.split("|")[0])
    if srcs:
        top_src = Counter(srcs).most_common(1)[0][0]
        print(f"    {cat:<22} → {top_src}")

print("\n" + "=" * 60)
print("[C. Abstention Analysis]")
print("=" * 60)

n_total   = len(rag_res)
n_abstain = rag_res["abstain_flag"].sum()
n_answer  = n_total - n_abstain

print(f"\n  Total tickets : {n_total}")
print(f"  Answered      : {n_answer}  ({100*n_answer/n_total:.0f}%)")
print(f"  Abstained     : {n_abstain}  ({100*n_abstain/n_total:.0f}%)")

abstained = rag_res[rag_res["abstain_flag"] == 1]
if len(abstained) > 0:
    print(f"\n  Abstained tickets:")
    for _, row in abstained.iterrows():
        print(f"    ticket {row['ticket_id']}: cat={row['predicted_category']:<22}"
              f"cat_conf={row['cat_confidence']:.2f}  ret_score={row['top_retrieval_score']:.3f}")

print(f"\n  Confidence score stats (answered tickets):")
answered = rag_res[rag_res["abstain_flag"] == 0]["confidence_score"]
print(f"    Mean  : {answered.mean():.3f}")
print(f"    Median: {answered.median():.3f}")
print(f"    Min   : {answered.min():.3f}")
print(f"    Max   : {answered.max():.3f}")

print("\n" + "=" * 60)
print("[D. Error Analysis]")
print("=" * 60)

low_conf = rag_res[rag_res["confidence_score"] < 0.3].sort_values("confidence_score")
print(f"\n  Low-confidence tickets (conf < 0.3): {len(low_conf)}")

eval_df = pd.read_csv("tickets_eval.csv")
eval_df["text"] = eval_df["subject"].fillna("") + " " + eval_df["message"].fillna("")
merged = rag_res.merge(eval_df[["ticket_id","subject","message"]], on="ticket_id")

if len(low_conf) > 0:
    lc_merged = merged[merged["confidence_score"] < 0.3].sort_values("confidence_score")
    for _, row in lc_merged.head(5).iterrows():
        print(f"\n  [ticket {row['ticket_id']}]")
        print(f"    Subject  : {row['subject']}")
        print(f"    Category : {row['predicted_category']}  Priority: {row['predicted_priority']}")
        print(f"    Conf     : {row['confidence_score']:.3f}  Ret score: {row['top_retrieval_score']:.3f}")
        print(f"    Sources  : {row['top_3_retrieved_sources']}")

print("\n  Source-category alignment issues:")
expected_src = {
    "billing": "billing",
    "technical_issue": "technical",
    "account_access": "account",
    "feature_request": "features",
    "general_query": "general",
}
mismatched = 0
for _, row in rag_res.iterrows():
    cat = row["predicted_category"]
    top_src = row["top_3_retrieved_sources"].split("|")[0]
    exp = expected_src.get(cat, "")
    if exp and exp not in top_src:
        mismatched += 1
print(f"    Tickets where top source doesn't match expected KB file: {mismatched}/{n_total}")

print("\n" + "=" * 60)
print("[E. Evaluation Summary]")
print("=" * 60)
cat_f1 = f1_score(y_cat, cat_cv, average="macro")
pri_f1 = f1_score(y_pri, pri_cv, average="macro")
print(f"""
  Classification (5-fold CV on 200 training tickets):
    Category F1-macro : {cat_f1:.3f}
    Priority F1-macro : {pri_f1:.3f}

  Retrieval (40 eval tickets):
    Mean retrieval score  : {scores.mean():.4f}
    Tickets with score>0.1: {(scores > 0.1).sum()} / {len(scores)}

  Abstention:
    Answered  : {n_answer} / {n_total}
    Abstained : {n_abstain} / {n_total}

  Key limitations:
    - KB docs are very short (~100 words each) → retrieval scores are low
    - No ground truth for eval → retrieval quality assessed heuristically
    - Abstention threshold tuned conservatively; can be adjusted per use-case
    - With semantic embeddings, retrieval quality would improve significantly
""")

print("[Part 4 complete]\n")
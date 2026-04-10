import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

train = pd.read_csv("tickets_train.csv")
eval_df = pd.read_csv("tickets_eval.csv")

def make_text(df):
    return (df["subject"].fillna("") + " " + df["message"].fillna("")).str.strip()

train["text"] = make_text(train)
eval_df["text"] = make_text(eval_df)

X_train = train["text"].tolist()
y_cat = train["category"].tolist()
y_pri = train["priority"].tolist()
X_eval = eval_df["text"].tolist()

print("=" * 60)
print("BASELINE: TF-IDF + Logistic Regression")
print("=" * 60)

def build_baseline(class_weight=None):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 1),
            max_features=3000,
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            C=1.0,
        )),
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_cat = build_baseline()
base_pri = build_baseline(class_weight="balanced")

cat_scores = cross_val_score(base_cat, X_train, y_cat, cv=cv, scoring="f1_macro")
pri_scores = cross_val_score(base_pri, X_train, y_pri, cv=cv, scoring="f1_macro")

print(f"\n[5-fold CV — Category]  F1-macro: {cat_scores.mean():.3f} ± {cat_scores.std():.3f}")
print(f"[5-fold CV — Priority]  F1-macro: {pri_scores.mean():.3f} ± {pri_scores.std():.3f}")

base_cat.fit(X_train, y_cat)
base_pri.fit(X_train, y_pri)

baseline_cat_preds = base_cat.predict(X_eval)
baseline_pri_preds = base_pri.predict(X_eval)
baseline_cat_proba = base_cat.predict_proba(X_eval).max(axis=1)
baseline_pri_proba = base_pri.predict_proba(X_eval).max(axis=1)

print("\n[Sample baseline predictions on eval]")
for i in range(5):
    print(f"  ticket {i}: cat={baseline_cat_preds[i]:<20} pri={baseline_pri_preds[i]:<8} conf={baseline_cat_proba[i]:.2f}")

print("\n" + "=" * 60)
print("IMPROVED: TF-IDF (1-2 grams, char) + LinearSVC + Calibration")
print("=" * 60)

URGENCY_WORDS = {
    "crash", "crashing", "broken", "error", "critical", "urgent",
    "cannot", "unable", "failed", "failure", "down", "outage",
    "immediately", "asap", "emergency", "lost", "data", "security",
    "breach", "hack", "charge", "overcharge", "wrong", "incorrect",
}

LOW_PRI_WORDS = {
    "feature", "request", "suggestion", "consider", "would", "could",
    "nice", "improve", "idea", "wish", "wondering", "question", "how",
    "general", "info", "learn",
}

class UrgencyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            words = set(text.lower().split())
            urgency_count = len(words & URGENCY_WORDS)
            low_pri_count = len(words & LOW_PRI_WORDS)
            has_exclaim = int("!" in text)
            word_count = len(text.split())
            rows.append([urgency_count, low_pri_count, has_exclaim, word_count])
        return np.array(rows, dtype=float)

def build_improved(class_weight=None):
    word_tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            stop_words="english",
            sublinear_tf=True,
            analyzer="word",
        )),
    ])
    char_tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(3, 5),
            max_features=3000,
            sublinear_tf=True,
            analyzer="char_wb",
        )),
    ])
    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
        ("urgency", UrgencyFeatures()),
    ])
    svc = LinearSVC(max_iter=2000, class_weight=class_weight, C=0.5)
    calibrated = CalibratedClassifierCV(svc, cv=5, method="sigmoid")
    return Pipeline([
        ("features", features),
        ("clf", calibrated),
    ])

imp_cat = build_improved()
imp_pri = build_improved(class_weight="balanced")

imp_cat_scores = cross_val_score(imp_cat, X_train, y_cat, cv=cv, scoring="f1_macro")
imp_pri_scores = cross_val_score(imp_pri, X_train, y_pri, cv=cv, scoring="f1_macro")

print(f"\n[5-fold CV — Category]  F1-macro: {imp_cat_scores.mean():.3f} ± {imp_cat_scores.std():.3f}")
print(f"[5-fold CV — Priority]  F1-macro: {imp_pri_scores.mean():.3f} ± {imp_pri_scores.std():.3f}")

imp_cat.fit(X_train, y_cat)
imp_pri.fit(X_train, y_pri)

improved_cat_preds = imp_cat.predict(X_eval)
improved_pri_preds = imp_pri.predict(X_eval)
improved_cat_proba = imp_cat.predict_proba(X_eval).max(axis=1)
improved_pri_proba = imp_pri.predict_proba(X_eval).max(axis=1)

print("\n[Sample improved predictions on eval]")
for i in range(5):
    print(f"  ticket {i}: cat={improved_cat_preds[i]:<20} pri={improved_pri_preds[i]:<8} conf={improved_cat_proba[i]:.2f}")

print("\n" + "=" * 60)
print("PER-CLASS REPORT (Improved model, 5-fold holdout)")
print("=" * 60)

cat_preds_cv = cross_val_predict(imp_cat, X_train, y_cat, cv=cv)
pri_preds_cv = cross_val_predict(imp_pri, X_train, y_pri, cv=cv)

print("\n[Category Classification Report]")
print(classification_report(y_cat, cat_preds_cv, digits=3))

print("[Priority Classification Report]")
print(classification_report(y_pri, pri_preds_cv, digits=3))

print("[Category Confusion Matrix]")
cats = sorted(set(y_cat))
cm = confusion_matrix(y_cat, cat_preds_cv, labels=cats)
header = f"{'':20}" + "".join(f"{c[:8]:>10}" for c in cats)
print(header)
for i, row_cat in enumerate(cats):
    row = f"{row_cat:<20}" + "".join(f"{v:>10}" for v in cm[i])
    print(row)

clf_results = pd.DataFrame({
    "ticket_id"          : eval_df["ticket_id"],
    "text"               : eval_df["text"],
    "predicted_category" : improved_cat_preds,
    "predicted_priority" : improved_pri_preds,
    "cat_confidence"     : improved_cat_proba.round(4),
    "pri_confidence"     : improved_pri_proba.round(4),
})
clf_results.to_csv("clf_results.csv", index=False)

print("\n[Saved clf_results.csv for Part 3]\n")
print("[Part 2 complete]\n")
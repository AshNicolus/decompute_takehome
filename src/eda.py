import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv("tickets_train.csv")
eval_df = pd.read_csv("tickets_eval.csv")

print("=" * 60)
print("PART 1: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\n[Dataset sizes]")
print(f"  Train : {len(train):>4} tickets")
print(f"  Eval  : {len(eval_df):>4} tickets")

print(f"\n[Train columns] : {list(train.columns)}")
print(f"[Eval  columns] : {list(eval_df.columns)}")

print("\n[Category distribution]")
cat_counts = train["category"].value_counts()
for cat, cnt in cat_counts.items():
    bar = "█" * cnt
    print(f"  {cat:<20} {cnt:>3}  {bar}")

print("\n[Priority distribution]")
pri_counts = train["priority"].value_counts()
for pri, cnt in pri_counts.items():
    bar = "█" * cnt
    print(f"  {pri:<10} {cnt:>3}  {bar}")

train["text"] = train["subject"].fillna("") + " " + train["message"].fillna("")
train["word_count"] = train["text"].apply(lambda x: len(x.split()))

print("\n[Message word count stats]")
print(train["word_count"].describe().to_string())

print("\n[Word count by category]")
print(train.groupby("category")["word_count"].mean().round(1).to_string())

print("\n[Word count by priority]")
print(train.groupby("priority")["word_count"].mean().round(1).to_string())

print("\n[Category × Priority crosstab]")
ct = pd.crosstab(train["category"], train["priority"])
print(ct.to_string())

print("\n[Top TF-IDF keywords per category]")
for cat in train["category"].unique():
    subset = train[train["category"] == cat]["text"].tolist()
    tfidf = TfidfVectorizer(max_features=8, stop_words="english")
    tfidf.fit(subset)
    top_words = list(tfidf.vocabulary_.keys())[:8]
    print(f"  {cat:<20}: {', '.join(top_words)}")

print("\n[Knowledge base files]")
kb_files = sorted(os.listdir("kb"))
for fname in kb_files:
    fpath = os.path.join("kb", fname)
    with open(fpath) as f:
        text = f.read()
    words = len(text.split())
    print(f"  {fname:<20} {words:>4} words")

print("\n[Key observations for modeling]")
observations = [
    "1. Well-balanced categories (~38-42 per class) → no class-weight issues",
    "2. Priority is slightly imbalanced (high=66, med=81, low=53) → use class_weight='balanced'",
    "3. Tickets are short (avg ~35 words) → TF-IDF works well; embeddings not strictly needed",
    "4. Category keywords are distinct → strong signal for classification",
    "5. KB docs are short summaries → sentence-level chunking appropriate",
    "6. 'general_query' may be hardest to distinguish from other classes",
    "7. Priority likely correlates with urgency words: 'urgent','crash','cannot','error'",
]
for obs in observations:
    print(f"  {obs}")

print("\n[Part 1 complete]\n")
# Technical Report: Customer Support Ticket AI System

---

## 1. Problem Framing & Assumptions

**Task.** Given a stream of customer support tickets, automatically (a) classify each ticket into a category and priority, and (b) generate a grounded draft response by retrieving relevant passages from a fixed knowledge base.

**Key assumptions made:**

- *Ticket text is sufficient signal.* Subject + message together provide enough context; no metadata (user ID, timestamp) is needed for the MVP.
- *KB is the ground truth for responses.* The system must never fabricate information. If the KB does not cover a topic, the system should abstain rather than hallucinate.
- *Priority is a derived label, not an inherent ticket property.* It correlates with urgency language ("crash", "cannot", "error") and category — technical/billing issues tend to be higher priority than feature requests.
- *Sentence-level chunking is appropriate.* Each KB document is ~100 words and covers one topic, so sentence-level chunks give finer retrieval granularity without losing context.

---

## 2. Part 1: Exploratory Data Analysis

**Dataset:** 200 labeled training tickets, 40 unlabeled eval tickets, 5 KB documents.

**Key findings:**

- **Category distribution is balanced** (~38–42 per class): account_access, billing, feature_request, general_query, technical_issue. No class-weighting needed for category.
- **Priority is mildly imbalanced** (high=66, medium=81, low=53). `class_weight='balanced'` used for the priority classifier.
- **Tickets are short** (mean 35 words, std 5). TF-IDF is well suited; dense embeddings are not strictly necessary at this scale.
- **Category-discriminating keywords are distinct**: billing→"charge/invoice/refund", technical→"crash/error/iOS", account→"password/login/reset", feature→"dark mode/integration/request", general→"export/CSV/plan".
- **Priority correlates with category** (from cross-tab): feature_request and general_query are almost exclusively low/medium; billing and technical are predominantly high. This means category is itself a strong feature for priority prediction.
- **KB documents are short summaries (~100 words each)** — retrieval scores will be low in absolute terms, but relative ranking should still be meaningful.

---

## 3. Part 2: Ticket Classification

### 3.1 Baseline — TF-IDF (unigrams) + Logistic Regression

- **Features:** TF-IDF with unigrams, 3000 features, English stopwords removed, sublinear TF scaling.
- **Model:** Logistic Regression (L2, C=1.0). Priority model uses `class_weight='balanced'`.
- **Result (5-fold CV):** Category F1-macro = **0.903**, Priority F1-macro = **0.718**.

The baseline is already strong because category keywords are highly discriminative in this dataset. Logistic Regression with TF-IDF is a robust, interpretable starting point.

### 3.2 Improved — TF-IDF (bigrams + char n-grams) + Calibrated LinearSVC

**Three improvements over baseline:**

1. **Bigrams** (word n-grams 1–2): captures phrases like "cannot log in", "dark mode", "incorrect charge" that carry more meaning than individual words.
2. **Character n-grams** (3–5 chars, char_wb): handles morphological variation ("crashing" ≈ "crashed"), misspellings, and partial word matches common in support tickets.
3. **Domain-informed urgency features**: 4 hand-crafted numeric features added alongside TF-IDF:
   - Count of urgency words (crash, error, cannot, overcharge, …)
   - Count of low-priority words (suggest, request, feature, wondering, …)
   - Presence of exclamation mark
   - Token count of message
   These features directly encode the business logic of priority assignment.
4. **Calibrated LinearSVC**: SVMs are well-suited for high-dimensional sparse text; calibration via Platt scaling gives reliable probability estimates needed for the confidence score.

**Result (5-fold CV):** Category F1-macro = **0.930** (+0.027), Priority F1-macro = **0.722** (+0.004).

### 3.3 Per-class analysis

| Category | Precision | Recall | F1 |
|---|---|---|---|
| account_access | 0.949 | 0.902 | 0.925 |
| billing | 0.930 | 0.976 | 0.952 |
| feature_request | 0.895 | 0.895 | 0.895 |
| general_query | 0.921 | 0.921 | 0.921 |
| technical_issue | 0.952 | 0.952 | 0.952 |

`feature_request` and `general_query` are the hardest to separate (both have overlap around "questions about the platform"). Priority classification is harder: `low` has the weakest recall (0.585) because feature requests labeled low-priority sometimes use urgent-sounding language.

---

## 4. Part 3: Retrieval + Response Generation

### 4.1 KB Chunking

Each KB `.txt` file is split on sentence boundaries. This yields 30 total chunks (6 per file). Sentence-level chunks allow finer-grained retrieval than returning the whole document.

### 4.2 TF-IDF Retrieval

A second TF-IDF vectorizer (bigrams, 8000 features) is fit on all 30 chunks. At query time, the ticket text is vectorized and cosine similarity is computed against all chunks to return top-3.

**Query augmentation:** The ticket text is concatenated with category-specific hint phrases (e.g., billing tickets get "billing subscription invoice payment refund" appended). This boosts recall for cases where the ticket uses informal language that doesn't lexically match the KB.

**Retrieval quality on eval (40 tickets):**
- All 40 tickets scored ≥ 0.10 (no complete misses)
- Mean retrieval score: 0.244
- 27/40 tickets (68%) scored ≥ 0.20
- Category → source alignment: 100% correct (billing tickets retrieve billing.txt, etc.)

### 4.3 Response Generation

Responses are template-based, grounding the answer in the top-2 retrieved sentences. The response:
- Acknowledges the ticket category
- Includes verbatim KB text (not fabricated)
- Cites source document(s)
- Encourages escalation if unresolved

This approach guarantees zero hallucination — every sentence in the draft comes from the KB.

### 4.4 Abstention Logic

The system abstains (sets `abstain_flag=1`, returns `"ABSTAIN: Insufficient information"`) when either:
- Top retrieval score < 0.05 (KB has no relevant content), or
- Classifier confidence < 0.30 (model is uncertain about category)

**Result on eval:** 1/40 tickets abstained (ticket 36: "Product arrived damaged" — an out-of-domain query not covered by the KB).

---

## 5. Part 4: Evaluation

### 5.1 Classification

| Model | Category F1 | Priority F1 | Category Acc | Priority Acc |
|---|---|---|---|---|
| Baseline (TF-IDF + LogReg) | 0.903 | 0.719 | 90.5% | 72.0% |
| **Improved (TF-IDF + SVM)** | **0.930** | **0.722** | **93.0%** | **72.5%** |

Category classification is strong. Priority prediction is harder — the 72% accuracy reflects genuine ambiguity in priority assignment (e.g., a general query about a billing issue could be medium or high).

### 5.2 Retrieval Quality

Retrieval is evaluated heuristically (no ground-truth relevance labels). Two proxies:

1. **Source alignment**: Does the top retrieved source match the expected KB file for the predicted category? Result: **100% alignment** (40/40 tickets retrieve the correct KB file as top source).
2. **Retrieval score distribution**: Mean 0.244, all tickets ≥ 0.10. While scores are low in absolute terms (expected for short KB docs), the relative ranking is consistent.

### 5.3 Abstention

- 1/40 tickets abstained. Ticket 36 ("Product arrived damaged") is a valid out-of-domain ticket — the KB covers software/SaaS support, not physical product complaints. The system correctly refused to answer rather than fabricate.

### 5.4 Error Analysis

- **Ticket 36** (abstained): Out-of-domain. The classifier assigned `general_query` with low confidence (0.27), and the retrieval matched `general.txt` but the content was irrelevant. Correct abstention.
- **Confusion matrix hotspots**: `account_access` ↔ `billing` (2 errors), `feature_request` ↔ `general_query` (2 errors). These pairs are semantically close and occasionally co-occur in real tickets.
- **Priority errors concentrate on `low`** class (recall 0.585) — feature requests sometimes use urgent vocabulary ("I really need this"), confusing the urgency detector.

---

## 6. Limitations & Next Steps

**Limitations:**
1. KB is very small (5 short docs, ~100 words each) — retrieval scores are inherently low. A richer KB would improve both scores and response quality.
2. Priority classification (72%) is the weakest component; it likely needs richer training data or human-defined priority rules.
3. Abstention threshold (0.05 retrieval, 0.30 classifier) is tuned conservatively on intuition, not held-out data — calibration study needed in production.
4. No semantic understanding — TF-IDF misses paraphrases ("can't get in" ≠ "unable to log in"). Sentence transformers (e.g., `all-MiniLM-L6-v2`) would improve this significantly.
5. Response generation is template-based — it won't handle multi-turn conversations or tickets requiring reasoning across multiple KB sections.

**Next steps:**
1. **Add semantic embeddings** (sentence-transformers + FAISS) for retrieval — straightforward drop-in improvement once internet access is available.
2. **LLM-based response generation** (Claude/Llama) for fluent, context-aware drafts with KB passages as grounding context.
3. **Active abstention calibration** — collect human judgments on borderline cases to set the abstention threshold empirically.
4. **Priority rule augmentation** — add SLA-level rules (e.g., any "data loss" or "security breach" → auto-high).
5. **Production monitoring** — track classifier confidence distributions and retrieval score distributions over time to detect drift.
6. **Feedback loop** — log agent overrides of predicted category/priority as new training signal.

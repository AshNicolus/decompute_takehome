# Customer Support Ticket AI System

An end-to-end AI system that classifies support tickets and generates grounded draft responses using a knowledge base — built entirely with open-source / offline tools (scikit-learn, numpy, pandas).

---

## Setup

### Requirements
- Python 3.8+
- scikit-learn ≥ 1.0
- pandas ≥ 1.3
- numpy ≥ 1.21

### Install dependencies
```bash
pip install scikit-learn pandas numpy
```

### Directory layout expected
```
support_system/
├── tickets_train.csv
├── tickets_eval.csv
├── kb/
│   ├── account.txt
│   ├── billing.txt
│   ├── features.txt
│   ├── general.txt
│   └── technical.txt
├── src/
│   ├── eda.py
│   ├── classification.py
│   ├── rag.py
│   ├── evaluation.py
│   └── predictions.py
├── requirements.txt
├── app.py
└── main.py
```

---

## How to reproduce results

### Option A — Run everything at once
```bash
cd support_system/
python run_all.py
```

### Option B — Run step by step
```bash
python eda.py           # EDA & observations
python classification.py # Classification models → clf_results.csv
python rag.py            # Retrieval + response generation → rag_results.csv
python evaluation.py     # Evaluation report
python predictions.py     # Final → predictions.csv
```

---

## Output files

| File | Description |
|------|-------------|
| `clf_results.csv` | Classifier predictions with per-class probabilities |
| `rag_results.csv` | Full pipeline output including retrieved sources and draft responses |
| `predictions.csv` | Final submission file in required schema |

### predictions.csv schema
```
ticket_id, predicted_category, predicted_priority,
top_3_retrieved_sources, draft_response,
confidence_score, abstain_flag
```

---

## External APIs / Models used
None. All computation is fully offline using highly optimized scikit-learn pipelines to strictly adhere to edge-device and low-VRAM memory constraints.

## Approximate cost
$0.00

---

## Key results (quick reference)

| Metric | Baseline | Improved |
|--------|----------|----------|
| Category F1-macro (5-fold CV) | 0.903 | **0.930** |
| Priority F1-macro (5-fold CV) | 0.719 | **0.722** |
| Abstention rate (eval) | — | 2.5% (1/40) |
| Mean confidence (answered) | — | 0.609 |

import subprocess, sys, os

scripts = [
    ("Part 1: EDA",             "src/eda.py"),
    ("Part 2: Classification",  "src/classification.py"),
    ("Part 3: RAG Pipeline",    "src/rag.py"),
    ("Part 4: Evaluation",      "src/evaluations.py"),
    ("Final: predictions.csv",  "src/predictions.py"),
]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

for label, script in scripts:
    print(f"\n{'='*60}")
    print(f"  Running {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed with code {result.returncode}")
        sys.exit(1)

print("\n" + "="*60)
print("  ALL PARTS COMPLETE")
print("  Output files: clf_results.csv, rag_results.csv, predictions.csv")
print("="*60)

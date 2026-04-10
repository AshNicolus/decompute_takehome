import pandas as pd

rag = pd.read_csv("rag_results.csv")

predictions = pd.DataFrame({
    "ticket_id": rag["ticket_id"],
    "predicted_category": rag["predicted_category"],
    "predicted_priority": rag["predicted_priority"],
    "top_3_retrieved_sources": rag["top_3_retrieved_sources"],
    "draft_response": rag["draft_response"],
    "confidence_score": rag["confidence_score"],
    "abstain_flag": rag["abstain_flag"],
})

predictions.to_csv("predictions.csv", index=False)

print(f"[predictions.csv saved — {len(predictions)} rows]")
print(predictions[["ticket_id", "predicted_category", "predicted_priority", "confidence_score", "abstain_flag"]].to_string(index=False))
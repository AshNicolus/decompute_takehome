import gradio as gr
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.classification import build_improved, X_train, y_cat, y_pri
from src.rag import retrieve, generate_response, CATEGORY_HINTS, ABSTAIN_THRESHOLD

print("Loading and training models... this takes about 5 seconds.")
cat_clf = build_improved()
pri_clf = build_improved(class_weight="balanced")

cat_clf.fit(X_train, y_cat)
pri_clf.fit(X_train, y_pri)

print("Models loaded successfully. Starting UI...")

def process_ticket(user_message):
    """
    The core logic that runs when a user submits a ticket in the UI.
    """
    if not user_message.strip():
        return "Please enter a ticket message.", "", "", ""

    category = cat_clf.predict([user_message])[0]
    priority = pri_clf.predict([user_message])[0]
    cat_conf = cat_clf.predict_proba([user_message]).max()
    
    hint = CATEGORY_HINTS.get(category, "")
    query = f"{user_message} {hint}"
    retrieved = retrieve(query, top_k=3)
    
    top_score = retrieved[0]["score"] if retrieved else 0.0

    should_abstain = (top_score < ABSTAIN_THRESHOLD) or (cat_conf < 0.30)
    
    if should_abstain:
        response = " **ABSTAIN:** Insufficient information in the Knowledge Base to provide a reliable answer."
        sources_str = "None"
    else:
        response = generate_response(user_message, category, retrieved)
        unique_sources = list(dict.fromkeys(r["source"] for r in retrieved))
        sources_str = ", ".join(unique_sources)

    cat_display = f"**{category.replace('_', ' ').title()}** (Conf: {cat_conf:.2f})"
    pri_display = f"**{priority.title()}**"
    
    return cat_display, pri_display, sources_str, response

with gr.Blocks(theme=gr.themes.Soft(), title="Decompute Support AI") as demo:
    gr.Markdown("# 🤖 Decompute Support AI System")
    gr.Markdown("Submit a mock customer support ticket below. The offline ML pipeline will classify it and draft a grounded response using the Knowledge Base.")
    
    with gr.Row():
        with gr.Column(scale=2):
            ticket_input = gr.Textbox(
                lines=5, 
                placeholder="E.g., I was double charged for my subscription this month. Please help!", 
                label="Incoming Support Ticket"
            )
            submit_btn = gr.Button("Process Ticket", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Triage Results")
            out_category = gr.Markdown("Category: *Pending*")
            out_priority = gr.Markdown("Priority: *Pending*")
            out_sources = gr.Markdown("Retrieved KB Sources: *Pending*")
            
    gr.Markdown("### 📝 Drafted Response")
    out_response = gr.Textbox(lines=8, label="", interactive=False)

    submit_btn.click(
        fn=process_ticket,
        inputs=ticket_input,
        outputs=[out_category, out_priority, out_sources, out_response]
    )

    ticket_input.submit(
        fn=process_ticket,
        inputs=ticket_input,
        outputs=[out_category, out_priority, out_sources, out_response]
    )

if __name__ == "__main__":
    demo.launch(share=False)
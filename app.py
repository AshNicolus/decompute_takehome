import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.classification import build_improved, X_train, y_cat, y_pri
from src.rag import retrieve, generate_response, CATEGORY_HINTS, ABSTAIN_THRESHOLD

st.set_page_config(page_title="Decompute AI Support", page_icon="🤖", layout="wide")

@st.cache_resource(show_spinner="Loading and training local models...")
def load_models():
    cat_clf = build_improved()
    pri_clf = build_improved(class_weight="balanced")
    
    cat_clf.fit(X_train, y_cat)
    pri_clf.fit(X_train, y_pri)
    return cat_clf, pri_clf

cat_clf, pri_clf = load_models()

st.title("🤖 Decompute Support AI System")
st.markdown("Submit a mock customer support ticket below. This pipeline runs **100% locally** using an offline ML architecture to classify the ticket and draft a grounded response.")
st.divider()

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📥 Incoming Ticket")
    user_message = st.text_area(
        "Enter support ticket message:",
        height=150,
        placeholder="e.g., I was double charged for my subscription this month. Please help!"
    )
    process_btn = st.button("Process Ticket", type="primary", use_container_width=True)

if process_btn and user_message.strip():
    with st.spinner("Analyzing ticket and retrieving KB sources..."):

        category = cat_clf.predict([user_message])[0]
        priority = pri_clf.predict([user_message])[0]
        cat_conf = cat_clf.predict_proba([user_message]).max()

        hint = CATEGORY_HINTS.get(category, "")
        query = f"{user_message} {hint}"
        retrieved = retrieve(query, top_k=3)
        
        top_score = retrieved[0]["score"] if retrieved else 0.0
       
        should_abstain = (top_score < ABSTAIN_THRESHOLD) or (cat_conf < 0.30)

        with col2:
            st.subheader("📊 Triage Results")
        
            display_cat = category.replace('_', ' ').title()
            display_pri = priority.title()
           
            m1, m2 = st.columns(2)
            m1.metric("Category", display_cat)
            m2.metric("Priority", display_pri)
            
            st.metric("Classifier Confidence", f"{cat_conf:.1%}")
            
            if should_abstain:
                st.error("**Retrieved Sources:** None (Confidence too low)")
            else:
                unique_sources = list(dict.fromkeys(r["source"] for r in retrieved))
                st.info(f"**Retrieved Sources:** {', '.join(unique_sources)}")
        
        st.divider()
        st.subheader("📝 Drafted Response")
        
        if should_abstain:
            st.warning("⚠️ **ABSTAIN:** Insufficient information in the Knowledge Base to provide a reliable answer. Escalating to human agent.")
        else:
            response = generate_response(user_message, category, retrieved)
            st.success(response)

elif process_btn:
    st.warning("Please enter a ticket message first.")
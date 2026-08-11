"""
Streamlit dashboard over the artifacts train.py produces:
  - model selector -> accuracy / F1 / per-class F1 / confusion matrix
  - misclassified examples for the selected model
  - live textbox: type a post, get a real-time prediction from the model

Run: streamlit run app.py   (after running train.py at least once)
"""
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import MODEL_LABELS, MODEL_NAMES, MODEL_PATH_TMPL, RESULTS_PATH, TEST_DATA_PATH, VECTORIZER_PATH

st.set_page_config(page_title="News Topic Classification", layout="wide")
st.title("News Topic Classification & Analytics Dashboard")
st.caption("20 Newsgroups · TF-IDF · Logistic Regression / Naive Bayes / Linear SVM")

try:
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    preds_df = pd.read_csv(TEST_DATA_PATH)
except FileNotFoundError:
    st.error("No results yet. Run `python train.py` first (needs internet to fetch 20 Newsgroups).")
    st.stop()

label_to_name = {MODEL_LABELS[n]: n for n in MODEL_NAMES}
choice_label = st.sidebar.selectbox("Model", list(label_to_name.keys()))
model_name = label_to_name[choice_label]
r = results[model_name]

st.sidebar.markdown("---")
st.sidebar.subheader("All models, quick compare")
compare_df = pd.DataFrame({
    "Model": [MODEL_LABELS[n] for n in MODEL_NAMES],
    "Accuracy": [results[n]["accuracy"] for n in MODEL_NAMES],
    "Macro F1": [results[n]["macro_f1"] for n in MODEL_NAMES],
}).set_index("Model")
st.sidebar.dataframe(compare_df.style.format("{:.3f}"))

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{r['accuracy']*100:.1f}%")
col2.metric("Macro F1", f"{r['macro_f1']:.3f}")
col3.metric("Weighted F1", f"{r['weighted_f1']:.3f}")

st.subheader("Confusion matrix")
labels = r["labels"]
fig_cm = px.imshow(
    r["confusion_matrix"], x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
    labels=dict(x="Predicted", y="Actual", color="Count"),
)
fig_cm.update_layout(height=450)
st.plotly_chart(fig_cm, width="stretch")

st.subheader("Per-class F1 score")
per_class = r["per_class"]
f1_df = pd.DataFrame({"category": list(per_class.keys()), "f1": [v["f1"] for v in per_class.values()]})
fig_f1 = px.bar(f1_df, x="category", y="f1", range_y=[0, 1])
st.plotly_chart(fig_f1, width="stretch")

st.subheader("Misclassified examples")
pred_col = f"pred_{model_name}"
wrong = preds_df[preds_df["true_label"] != preds_df[pred_col]][["text", "true_label", pred_col]]
wrong = wrong.rename(columns={pred_col: "predicted_label"})
n_show = st.slider("How many examples to show", 3, 20, 5)
for _, row in wrong.head(n_show).iterrows():
    with st.container(border=True):
        st.markdown(f"**True:** `{row['true_label']}`  &nbsp;&nbsp; **Predicted:** `{row['predicted_label']}`")
        st.write(row["text"][:400] + ("..." if len(row["text"]) > 400 else ""))

st.divider()
st.subheader("Try it yourself")
user_text = st.text_area("Paste or write a post, then predict its category:", height=120)
if st.button("Predict") and user_text.strip():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH_TMPL.format(name=model_name))
    X = vectorizer.transform([user_text])
    pred_idx = model.predict(X)[0]
    st.success(f"Predicted category: **{labels[pred_idx]}**")

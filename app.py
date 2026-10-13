import streamlit as st
from recommend import Recommender

st.set_page_config(page_title="MeSH Keyword Recommender", layout="wide")
@st.cache_resource
def load(): return Recommender()
rec = load()

st.title("Keyword Recommendation for Clinical Trial Search")
st.caption("Collaborative filtering over MeSH terms (ClinicalTrials.gov). Paste a query; get keywords you may have missed.")
examples = {"Breast cancer": "A phase 3 study of chemotherapy and trastuzumab in women with HER2 positive breast cancer",
            "Diabetes": "Insulin therapy versus metformin in adults with type 2 diabetes and obesity",
            "Heart": "Statin treatment for patients with hypertension after myocardial infarction"}
c = st.columns(len(examples))
for col, (k, v) in zip(c, examples.items()):
    if col.button(k): st.session_state["q"] = v
q = st.text_area("Query / trial description", key="q", height=140)
k = st.slider("Number of recommendations", 1, 10, 5)
if st.button("Recommend", type="primary") and q.strip():
    found, recs = rec.recommend(q, k)
    l, r = st.columns(2)
    l.subheader("MeSH keywords found in your query")
    l.write(", ".join(f"**{t}**" for t in found) if found else "None found - try more medical terms.")
    r.subheader("Recommended additional keywords")
    for t, sc in recs: r.write(f"**{t}** (score {sc:.2f})")
    if found and not recs: r.write("No confident recommendations.")

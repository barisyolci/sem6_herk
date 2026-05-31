import csv
import io
import os
from typing import Dict, List
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from text_utils import _from_bytes, load_uploaded
from metrics import run_eval_input
from llm_evaluator import LLMJudge
from semantic_engine import SemanticScorer
from vector_scorer import highlight_key_terms as tfidf_explain
from config import _tgt_file, _lbl_file, _res_dir, _out_path

load_dotenv()

@st.cache_resource
def _load_sem(model_name="all-MiniLM-L6-v2"):
    return SemanticScorer(model_name=model_name)

def _parse_labels(raw: bytes) -> Dict[str, str]:
    try:
        txt = raw.decode("utf-8", errors="ignore")
        rdr = csv.DictReader(io.StringIO(txt))
        m = {}
        for row in rdr:
            cid = row.get("candidate_id", "").strip()
            rel = row.get("relevance", "").strip()
            if cid and rel:
                m[cid] = rel
        return m
    except:
        return {}

def _init_state():
    defs = {
        "tgt": "",
        "cids": [],
        "ctexts": [],
        "truth": {},
        "ranks": {},
        "summ": pd.DataFrame(),
        "done": False,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _get_text(uf):
    if uf is None:
        return ""
    return _from_bytes(uf.name, uf.getvalue()).strip()

def main():
    st.set_page_config(page_title="Candidate Matching System", layout="wide")

    css = """
    <style>
    .main { background-color: #f8f9fa; padding-top: 2rem; }
    .stTitle { color: #0f172a; font-family: system-ui, sans-serif; font-weight: 700; }
    .stSubheader { color: #334155; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; }
    .css-1lcbmhc { background-color: #ffffff; border-radius: 8px; padding: 1rem; border: 1px solid #e2e8f0; }
    .stButton>button {
        background-color: #2563eb;
        color: #ffffff;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    .stDataFrame { border-radius: 8px; border: 1px solid #e2e8f0; margin-top: 1rem; }
    hr { border-top: 1px solid #cbd5e1; margin: 2rem 0; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    _init_state()

    st.title("Candidate Matching System")
    st.write("Evaluate alignment between target roles and candidate profiles using vector and semantic scoring.")

    st.subheader("Target Role Description")
    col1, col2 = st.columns(2)
    with col1:
        up_tgt = st.file_uploader("Upload target role", type=["txt"], label_visibility="collapsed")
        if up_tgt:
            st.session_state.tgt = _get_text(up_tgt)
        elif not st.session_state.tgt and _tgt_file.exists():
            st.session_state.tgt = _tgt_file.read_text(encoding="utf-8").strip()
    with col2:
        if st.session_state.tgt:
            st.text_area("Preview", st.session_state.tgt, height=150, disabled=True, label_visibility="collapsed")

    st.markdown("---")
    st.subheader("Candidate Profiles")
    col3, col4 = st.columns(2)
    with col3:
        up_res = st.file_uploader("Upload resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed")
        if up_res:
            ids, texts = load_uploaded([(f.name, f.getvalue()) for f in up_res])
            st.session_state.cids = ids
            st.session_state.ctexts = texts
        elif not st.session_state.cids and _res_dir.exists():
            ids, texts = load_uploaded([(p.name, p.read_bytes()) for p in _res_dir.glob("*") if p.is_file()])
            st.session_state.cids = ids
            st.session_state.ctexts = texts
    with col4:
        if st.session_state.cids:
            st.write(f"Loaded candidates: {len(st.session_state.cids)}")

    st.markdown("---")
    st.subheader("Ground Truth Labels")
    col5, col6 = st.columns(2)
    with col5:
        up_lbl = st.file_uploader("Upload labels CSV", type=["csv"], label_visibility="collapsed")
        if up_lbl:
            st.session_state.truth = _parse_labels(up_lbl.getvalue())
        elif not st.session_state.truth and _lbl_file.exists():
            st.session_state.truth = _parse_labels(_lbl_file.read_bytes())
    with col6:
        if st.session_state.truth:
            st.write(f"Mapped labels: {len(st.session_state.truth)}")

    st.markdown("---")
    st.subheader("Scoring Configuration")
    col7, col8 = st.columns(2)
    with col7:
        use_tfidf = st.checkbox("Vector Scoring (TF-IDF)", value=True, disabled=True)
    with col8:
        use_sem = st.checkbox("Semantic Scoring (Embeddings)", value=True)

    st.markdown("---")
    if st.button("Run Analysis", type="primary"):
        if not st.session_state.tgt or not st.session_state.cids:
            st.error("Upload target role and candidate files first.")
        else:
            with st.spinner("Processing..."):
                sem = _load_sem() if use_sem else None
                rks, summ = run_eval_input(
                    tgt=st.session_state.tgt,
                    cids=st.session_state.cids,
                    ctexts=st.session_state.ctexts,
                    truth=st.session_state.truth,
                    use_llm=False,
                    sem_mod=sem,
                    llm_mod=None,
                    out_dir=str(_out_path),
                    save=True
                )
                st.session_state.ranks = rks
                st.session_state.summ = summ
                st.session_state.done = True

    if st.session_state.done:
        st.markdown("---")
        st.subheader("Evaluation Results")
        st.dataframe(st.session_state.summ, use_container_width=True, hide_index=True)

        st.subheader("Detailed Rankings")
        methods = list(st.session_state.ranks.keys())
        tabs = st.tabs(methods)
        for tab, method in zip(tabs, methods):
            with tab:
                df = st.session_state.ranks[method]
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.subheader("Candidate Details")
                if st.session_state.cids:
                    sel = st.selectbox("Select candidate", st.session_state.cids, key=f"sel_{method}")
                    idx = st.session_state.cids.index(sel)
                    tgt = st.session_state.tgt
                    ct = st.session_state.ctexts[idx]
                    sc = df[df["candidate_id"] == sel]["match_score"].values[0]
                    st.write(f"Score: {sc:.4f}")

                    if method == "tfidf":
                        s, terms = tfidf_explain(tgt, ct, k=10)
                        if terms:
                            st.write("Top terms:")
                            for term, weight in terms:
                                st.write(f"{term}: {weight:.4f}")
                        else:
                            st.write("No significant terms.")
                    elif method == "semantic":
                        sem = _load_sem()
                        s, shared = sem.explain_match(tgt, ct, top_n=10)
                        if shared:
                            st.write("Top keywords:")
                            for kw, cnt in shared:
                                st.write(f"{kw}: {cnt}")
                        else:
                            st.write("No shared keywords.")
                    else:
                        st.write("Baseline score calculation.")

if __name__ == "__main__":
    main()
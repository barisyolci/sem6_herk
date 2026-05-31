import csv
import io
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from text_utils import extract_from_bytes, load_uploaded
from metrics import generate_ranking, execute_evaluation_from_input, calculate_metrics
from llm_evaluator import LLMJudge
from semantic_engine import SemanticScorer
from vector_scorer import get_keyword_contributions as tfidf_explain
from vector_scorer import compute_match_scores as tfidf_compute
from config import ASSETS_DIR, RESULTS_DIR, DEFAULT_TARGET_FILE, DEFAULT_LABELS_FILE, DEFAULT_RESUMES_DIR

load_dotenv()

@st.cache_resource
def _instantiate_semantic_engine(model_name="all-MiniLM-L6-v2"):
    return SemanticScorer(model_identifier=model_name)

@st.cache_resource
def _instantiate_llm_evaluator(use_api=True, model_id="openai/gpt-oss-120b", effort="medium"):
    return LLMJudge(use_api=use_api, model_id=model_id, effort_level=effort)

def _parse_ground_truth(raw_bytes):
    try:
        decoded = raw_bytes.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(decoded))
        mapping = {}
        for row in reader:
            cid = row.get("candidate_id", "").strip()
            relevance = row.get("relevance", "").strip()
            if cid and relevance:
                mapping[cid] = relevance
        return mapping
    except Exception:
        return {}

def _load_default_assets():
    target = DEFAULT_TARGET_FILE.read_text(encoding="utf-8").strip()
    ids, texts = load_uploaded(
        [(p.name, p.read_bytes()) for p in DEFAULT_RESUMES_DIR.glob("*") if p.is_file()]
    )
    truth = _parse_ground_truth(DEFAULT_LABELS_FILE.read_bytes())
    return target, ids, texts, truth

def _initialize_session_state():
    defaults = {
        "target_desc": "",
        "candidate_ids": [],
        "candidate_texts": [],
        "ground_truth": {},
        "ranking_tables": {},
        "summary_metrics": pd.DataFrame(),
        "is_evaluated": False,
        "selected_candidate": None,
        "candidate_scores": {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def _render_summary(results_df):
    st.subheader("Evaluation Summary (Percentages)")
    display = results_df.copy()
    pct_cols = ["hit_rate_k1", "hit_rate_k3", "hit_rate_k5", "precision_k3"]
    display[pct_cols] = display[pct_cols].round(2)
    display["avg_pos_relevant"] = display["avg_pos_relevant"].round(2)
    st.dataframe(display, use_container_width=True)

def _extract_upload_text(uploaded_file):
    if uploaded_file is None:
        return ""
    return extract_from_bytes(uploaded_file.name, uploaded_file.getvalue()).strip()

def main():
    st.set_page_config(page_title="Candidate Role Alignment", layout="wide")
    st.title("Candidate Role Alignment Dashboard")
    st.caption("Evaluate and compare scoring methods for candidate-role matching.")

    _initialize_session_state()

    with st.sidebar:
        st.header("Configuration")
        st.subheader("Scoring Methods")
        use_tfidf = st.checkbox("TF-IDF Vectorizer", value=True, disabled=True)
        use_semantic = st.checkbox("Semantic Embeddings", value=True)
        use_llm = st.checkbox("LLM Evaluation", value=False)

        if use_llm:
            st.divider()
            st.subheader("LLM Settings")
            groq_key = st.text_input("GROQ_API_KEY", type="password", value=os.getenv("GROQ_API_KEY", ""))
            llm_model = st.selectbox("Model", ["openai/gpt-oss-120b", "meta-llama/llama-3.3-70b-versatile", "mistralai/mixtral-8x7b-32768"], index=0)
            llm_effort = st.select_slider("Reasoning Effort", options=["low", "medium", "high"], value="medium")
            st.info("Without an API key, a deterministic mock scorer will be used.")

        st.divider()
        if st.button("Reset State", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()
        if st.button("Run Evaluation", type="primary"):
            target_text = st.session_state.target_desc
            cand_ids = st.session_state.candidate_ids
            cand_texts = st.session_state.candidate_texts
            truth = st.session_state.ground_truth

            if not target_text or not cand_ids:
                st.sidebar.error("Upload target role and candidate files first.")
            else:
                with st.spinner("Processing evaluation..."):
                    semantic_eng = None
                    llm_judge = None
                    if use_semantic:
                        semantic_eng = _instantiate_semantic_engine()
                    if use_llm:
                        llm_judge = _instantiate_llm_evaluator(
                            use_api=bool(groq_key.strip()),
                            model_id=llm_model,
                            effort=llm_effort
                        )

                    rankings, summary = execute_evaluation_from_input(
                        target_text=target_text,
                        candidate_ids=cand_ids,
                        candidate_texts=cand_texts,
                        ground_truth=truth,
                        enable_llm=use_llm,
                        semantic_model=semantic_eng,
                        llm_model=llm_judge,
                        output_dir=str(RESULTS_DIR),
                        persist_results=True
                    )

                    st.session_state.ranking_tables = rankings
                    st.session_state.summary_metrics = summary
                    st.session_state.is_evaluated = True

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Role Description")
        uploaded_target = st.file_uploader("Upload target role (.txt)", type=["txt"], key="target_up")
        if uploaded_target:
            st.session_state.target_desc = _extract_upload_text(uploaded_target)
        elif not st.session_state.target_desc:
            if DEFAULT_TARGET_FILE.exists():
                st.session_state.target_desc = DEFAULT_TARGET_FILE.read_text(encoding="utf-8").strip()
        st.text_area("Preview", st.session_state.target_desc, height=200, disabled=True, key="target_preview")

    with col2:
        st.subheader("Candidate Resumes")
        uploaded_resumes = st.file_uploader("Upload resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True, key="resumes_up")
        if uploaded_resumes:
            ids, texts = load_uploaded([(f.name, f.getvalue()) for f in uploaded_resumes])
            st.session_state.candidate_ids = ids
            st.session_state.candidate_texts = texts
        elif not st.session_state.candidate_ids:
            if DEFAULT_RESUMES_DIR.exists():
                ids, texts = load_uploaded(
                    [(p.name, p.read_bytes()) for p in DEFAULT_RESUMES_DIR.glob("*") if p.is_file()]
                )
                st.session_state.candidate_ids = ids
                st.session_state.candidate_texts = texts
        st.write(f"Loaded candidates: {len(st.session_state.candidate_ids)}")

    st.subheader("Ground Truth Labels")
    uploaded_labels = st.file_uploader("Upload labels (.csv)", type=["csv"], key="labels_up")
    if uploaded_labels:
        st.session_state.ground_truth = _parse_ground_truth(uploaded_labels.getvalue())
    elif not st.session_state.ground_truth:
        if DEFAULT_LABELS_FILE.exists():
            st.session_state.ground_truth = _parse_ground_truth(DEFAULT_LABELS_FILE.read_bytes())
    st.write(f"Labels mapped: {len(st.session_state.ground_truth)}")

    if st.session_state.is_evaluated:
        st.divider()
        st.subheader("Evaluation Results")
        _render_summary(st.session_state.summary_metrics)

        st.subheader("Detailed Rankings")
        method_tabs = st.tabs([m.capitalize() for m in st.session_state.ranking_tables.keys()])
        for tab, method_name in zip(method_tabs, st.session_state.ranking_tables.keys()):
            with tab:
                df = st.session_state.ranking_tables[method_name]
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.subheader("Candidate Breakdown")
                if st.session_state.candidate_ids:
                    selected = st.selectbox("Select candidate for detailed view", st.session_state.candidate_ids, key=f"sel_{method_name}")
                    idx = st.session_state.candidate_ids.index(selected)
                    target_text = st.session_state.target_desc
                    cand_text = st.session_state.candidate_texts[idx]
                    score = df[df["candidate_id"] == selected]["match_score"].values[0]
                    st.write(f"Score: {score:.4f}")

                    if method_name == "tfidf":
                        sim, terms = tfidf_explain(target_text, cand_text, max_terms=10)
                        if terms:
                            st.write("Top contributing terms:")
                            for term, weight in terms:
                                st.text(f"{term}: {weight:.4f}")
                        else:
                            st.text("No significant term overlap.")
                    elif method_name == "semantic":
                        semantic_eng = _instantiate_semantic_engine()
                        sim, shared = semantic_eng.explain_match(target_text, cand_text, top_terms=10)
                        if shared:
                            st.write("Top shared keywords:")
                            for kw, count in shared:
                                st.text(f"{kw}: {count}")
                        else:
                            st.text("No shared keywords detected.")
                    else:
                        llm_judge = _instantiate_llm_evaluator(use_api=False)
                        rationale = llm_judge.provide_rationale(target_text, cand_text, include_meta=False)
                        st.write("LLM Rationale:")
                        st.text(rationale)

if __name__ == "__main__":
    main()
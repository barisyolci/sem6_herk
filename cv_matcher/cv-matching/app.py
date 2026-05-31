import csv
import io
import os
from typing import Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from data_loader import extract_text_from_bytes, load_uploaded_files
from evaluate import (
    load_job_description,
    rank_scores,
    run_benchmark,
    run_benchmark_from_texts,
)
from llm_model import LLMScorer
from sbert_model import SBERTScorer
from tfidf_model import explain_pair as tfidf_explain_pair
from tfidf_model import score_cvs as tfidf_score


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
load_dotenv()


@st.cache_resource
def get_sbert_scorer() -> SBERTScorer:
    return SBERTScorer()


@st.cache_resource
def get_llm_scorer(use_groq: bool, model_name: str, reasoning_effort: str) -> LLMScorer:
    return LLMScorer(use_groq=use_groq, model_name=model_name, reasoning_effort=reasoning_effort)


def parse_labels_csv(data: bytes) -> Dict[str, str]:
    text = data.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    labels: Dict[str, str] = {}
    for row in reader:
        cv_id = row.get("cv_id", "").strip()
        label = row.get("label", "").strip()
        if cv_id:
            labels[cv_id] = label
    return labels


def display_benchmark_results(results_df: pd.DataFrame) -> None:
    st.subheader("Benchmark Results (Percentages)")
    display_df = results_df.copy()
    pct_cols = ["top1_accuracy", "top3_accuracy", "top5_accuracy", "precision_at_3"]
    display_df[pct_cols] = display_df[pct_cols].round(2)
    display_df["mean_rank_relevant"] = display_df["mean_rank_relevant"].round(2)
    st.dataframe(display_df, use_container_width=True)


def get_upload_text(file) -> str:
    if file is None:
        return ""
    return extract_text_from_bytes(file.name, file.getvalue()).strip()


def main() -> None:
    st.set_page_config(page_title="CV Matching Benchmark", layout="wide")
    st.title("CV Matching Benchmark")
    st.write("Compare TF-IDF, SBERT, and Groq LLM scoring on a small CV dataset.")

    mode = st.radio("Mode", ["Benchmark (sample dataset)", "Upload files"], horizontal=True)

    api_key_present = bool(os.getenv("GROQ_API_KEY"))
    use_groq = st.checkbox(
        "Use Groq LLM scoring (requires GROQ_API_KEY)",
        value=api_key_present,
        help="If disabled or no API key is set, a deterministic mock scorer is used.",
    )
    if use_groq and not api_key_present:
        st.warning("GROQ_API_KEY is not set. The LLM scorer will fall back to the mock implementation.")
    elif not use_groq:
        st.info("LLM scoring uses the deterministic mock implementation.")

    with st.expander("LLM settings", expanded=False):
        model_options = ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]
        model_choice = st.selectbox("Model", model_options, index=0)
        custom_model = st.text_input("Custom model (optional)")
        model_name = custom_model.strip() or model_choice
        reasoning_effort = st.selectbox(
            "Reasoning effort",
            ["low", "medium", "high"],
            index=1,
            help="Controls internal reasoning depth; output is still a single number.",
        )
        show_llm_reason = st.checkbox(
            "Show LLM rationale in 1:1 scoring",
            value=False,
            help="Adds a brief explanation for the LLM score (1:1 mode only).",
        )
        show_tfidf_reason = st.checkbox(
            "Show TF-IDF explanation in 1:1 scoring",
            value=False,
            help="Displays top TF-IDF term overlaps for the 1:1 score.",
        )
        show_sbert_reason = st.checkbox(
            "Show SBERT explanation in 1:1 scoring",
            value=False,
            help="Displays top shared keywords alongside the SBERT similarity.",
        )
    st.caption(f"LLM model: {model_name} | reasoning effort: {reasoning_effort}")

    if mode == "Benchmark (sample dataset)":
        job_files = ["job_description.txt"]
        job_file = st.selectbox("Select job description", job_files)
        job_path = os.path.join(DATA_DIR, job_file)
        job_description = load_job_description(job_path)

        with st.expander("View job description", expanded=False):
            st.text(job_description)

        if st.button("Run benchmark"):
            with st.spinner("Scoring CVs and computing benchmarks..."):
                rankings, results_df = run_benchmark(
                    DATA_DIR,
                    use_groq=use_groq,
                    sbert_scorer=get_sbert_scorer(),
                    llm_scorer=get_llm_scorer(use_groq, model_name, reasoning_effort),
                )

            display_benchmark_results(results_df)

            st.subheader("Rankings")
            for model_name, ranking in rankings.items():
                st.markdown(f"**{model_name.upper()}**")
                st.dataframe(ranking, use_container_width=True)
        return

    comparison_type = st.radio(
        "Comparison type",
        ["One job vs many CVs", "One job vs one CV"],
        horizontal=True,
    )

    job_file = st.file_uploader(
        "Upload job description (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
    )

    if comparison_type == "One job vs many CVs":
        cv_files = st.file_uploader(
            "Upload CV files (.txt, .pdf, .docx)",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
        )
        labels_file = st.file_uploader(
            "Optional labels CSV (cv_id,label)",
            type=["csv"],
        )
        save_results = st.checkbox("Save results to results/benchmark_results.csv", value=False)

        if st.button("Run upload benchmark"):
            if job_file is None or not cv_files:
                st.error("Please upload a job description and at least one CV file.")
                return

            job_text = get_upload_text(job_file)
            cv_ids, cv_texts = load_uploaded_files(
                [(file.name, file.getvalue()) for file in cv_files]
            )

            if not job_text:
                st.error("Job description text could not be extracted.")
                return

            if labels_file is not None:
                labels = parse_labels_csv(labels_file.getvalue())
                with st.spinner("Scoring CVs and computing benchmarks..."):
                    rankings, results_df = run_benchmark_from_texts(
                        job_text,
                        cv_ids,
                        cv_texts,
                        labels,
                        use_groq=use_groq,
                        sbert_scorer=get_sbert_scorer(),
                        llm_scorer=get_llm_scorer(use_groq, model_name, reasoning_effort),
                        results_dir=os.path.join(os.path.dirname(__file__), "results"),
                        save_results=save_results,
                    )

                display_benchmark_results(results_df)
            else:
                sbert_scorer = get_sbert_scorer()
                llm_scorer = get_llm_scorer(use_groq, model_name, reasoning_effort)
                scores = {
                    "tfidf": tfidf_score(job_text, cv_texts),
                    "sbert": sbert_scorer.score_cvs(job_text, cv_texts),
                    "llm": llm_scorer.score_cvs(job_text, cv_texts),
                }
                rankings = {
                    name: rank_scores(cv_ids, score_list, {})
                    for name, score_list in scores.items()
                }
                st.info("Upload labels.csv to compute benchmark percentages.")

            st.subheader("Rankings")
            for model_name, ranking in rankings.items():
                st.markdown(f"**{model_name.upper()}**")
                st.dataframe(ranking, use_container_width=True)
        return

    cv_file = st.file_uploader(
        "Upload a single CV (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
    )

    if st.button("Run 1:1 scoring"):
        if job_file is None or cv_file is None:
            st.error("Please upload a job description and one CV file.")
            return

        job_text = get_upload_text(job_file)
        cv_text = get_upload_text(cv_file)

        if not job_text or not cv_text:
            st.error("Text extraction failed for the uploaded files.")
            return

        sbert_scorer = get_sbert_scorer()
        llm_scorer = get_llm_scorer(use_groq, model_name, reasoning_effort)
        scores = [
            ("TF-IDF", tfidf_score(job_text, [cv_text])[0]),
            ("SBERT", sbert_scorer.score_cvs(job_text, [cv_text])[0]),
            ("LLM", llm_scorer.score_cvs(job_text, [cv_text])[0]),
        ]
        results_df = pd.DataFrame(
            {
                "model": [item[0] for item in scores],
                "score_0_to_1": [round(item[1], 4) for item in scores],
            }
        )
        st.subheader("1:1 Scores")
        st.dataframe(results_df, use_container_width=True)
        if show_tfidf_reason:
            st.subheader("TF-IDF Explanation")
            similarity, terms = tfidf_explain_pair(job_text, cv_text, top_n=10)
            st.write(f"Cosine similarity: {similarity:.4f}")
            if terms:
                st.dataframe(
                    {"term": [t[0] for t in terms], "contribution": [round(t[1], 6) for t in terms]},
                    use_container_width=True,
                )
            else:
                st.write("No overlapping TF-IDF terms found.")
        if show_sbert_reason:
            st.subheader("SBERT Explanation")
            normalized_similarity, keywords = sbert_scorer.explain_pair(job_text, cv_text, top_n=10)
            st.write(f"Normalized cosine similarity: {normalized_similarity:.4f}")
            if keywords:
                st.dataframe(
                    {"keyword": [k[0] for k in keywords], "count": [k[1] for k in keywords]},
                    use_container_width=True,
                )
            else:
                st.write("No shared keywords found.")
        if show_llm_reason:
            st.subheader("LLM Rationale")
            with st.spinner("Generating rationale..."):
                rationale, finish_reason = llm_scorer.explain_pair(
                    job_text, cv_text, return_meta=True
                )
            st.markdown(rationale)
            if finish_reason == "length":
                st.warning(
                    "Rationale may be truncated by the model output limit."
                )


if __name__ == "__main__":
    main()

"""Streamlit demo for the session-based recommender."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from recsys.serving.predictor import Predictor


@st.cache_resource
def load_predictor(model_path: str) -> Predictor:
    return Predictor.from_path(model_path)


def main() -> None:
    st.set_page_config(page_title="RecSys Demo", page_icon=":bar_chart:")
    st.title("Session Recommendation Demo")

    model_path = st.text_input("Model artifact", "models/trained/latest/model.json")
    item_sequence_raw = st.text_input("Item sequence", "101,205,330")
    top_k = st.number_input("Top K", min_value=1, max_value=50, value=10)

    if st.button("Recommend"):
        item_sequence = [int(part.strip()) for part in item_sequence_raw.split(",") if part.strip()]
        if not Path(model_path).exists():
            st.error(f"Model artifact not found: {model_path}")
            return
        predictor = load_predictor(model_path)
        recommendations = predictor.get_recommendations(item_sequence, top_k=int(top_k))
        st.write(recommendations)


if __name__ == "__main__":
    main()

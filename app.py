import os
import shutil
import streamlit as st
import pandas as pd
from pathlib import Path

from pypekit import Repository, CachedExecutor
from algo_repo import (
    IrisLoader,
    MinMaxScaler,
    StandardScaler,
    PCA,
    LogisticRegression,
    RandomForestClassifier,
    SVC,
)


def build_and_evaluate_pipelines(algo_list):
    if not os.path.exists("./cache"):
        os.makedirs("./cache")
    repository = Repository(algo_list)
    pipelines = repository.build_pipelines()
    executor = CachedExecutor(cache_dir="./cache", pipelines=pipelines)
    results = executor.run()

    for result in results:
        df = pd.read_csv(result["output_path"])
        result["accuracy"] = (df["y_pred"] == df["target"]).mean()
    
    shutil.rmtree("./cache")
    return results


st.set_page_config(page_title="Pipeline Builder", page_icon="🧩", layout="wide")
st.title("🧩 Pipeline Builder")
st.markdown(
    """
    Select the building‑blocks for your pipeline.  When you
    click **Run Pipelines**, every valid ordering will be generated, executed
    (with caching), and the accuracy of each resulting model will be displayed.
    """
)

ALGO_FACTORY = {
    "iris_loader": IrisLoader,
    "minmax_scaler": MinMaxScaler,
    "standard_scaler": StandardScaler,
    "pca": PCA,
    "logistic_regression": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier,
    "svc": SVC,
}

ALL_ALGOS = list(ALGO_FACTORY.keys())

st.sidebar.header("⚙️ Configure")

selected_names = st.sidebar.multiselect(
    "Algorithms to include", ALL_ALGOS, default=None
)

run_button = st.sidebar.button("Run Pipelines", type="primary")


if run_button:
    if not selected_names:
        st.warning("Please select at least one algorithm to proceed.")
        st.stop()

    selected_algos = []
    for name in selected_names:
        cls = ALGO_FACTORY[name]
        if name == "pca":
            instance = cls(n_components=2)
        else:
            instance = cls()
        selected_algos.append((name, instance))

    with st.spinner("Building & evaluating pipelines…"):
        results = build_and_evaluate_pipelines(selected_algos)

    records = [
        {
            "Pipeline Tasks": " → ".join(r.get("tasks", [])),
            "Accuracy": round(r.get("accuracy", float("nan")), 4),
        }
        for r in results
    ]
    result_df = pd.DataFrame(records)

    st.success(f"Finished {len(result_df)} pipelines! ✨")
    st.dataframe(result_df, use_container_width=True)

else:
    st.info("Pick your algorithms in the sidebar and hit **Run Pipelines**.")


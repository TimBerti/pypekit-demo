import streamlit as st
import pandas as pd

from pypekit import Repository, CachedExecutor
from algo_repo import ALGORITHMS


def build_and_evaluate_pipelines(algo_list):
    repository = Repository(set(algo_list))
    try:
        repository.build_tree()
        pipelines = repository.build_pipelines()
        if len(pipelines) == 1:
            if len(pipelines[0].tasks) == 1:
                raise ValueError("No valid pipeline found.")
    except ValueError:
        st.error("No valid pipeline found.")
        return None, None
    executor = CachedExecutor(pipelines)
    results = executor.run()
    return repository, results

meltpoolnet_df = ALGORITHMS["Data Loader"]["MeltPoolNet Loader"]().run()
iris_df = ALGORITHMS["Data Loader"]["Iris Loader"]().run()
wine_df = ALGORITHMS["Data Loader"]["Wine Loader"]().run()

st.title("Pipeline Synthesis Demo")

st.write("### Datasets")
tab1, tab2, tab3 = st.tabs(["MeltPoolNet Dataset", "Iris Dataset", "Wine Dataset"])
with tab1:
    st.dataframe(meltpoolnet_df[:5])
with tab2:
    st.dataframe(iris_df[:5])
with tab3:
    st.dataframe(wine_df[:5])

st.write("### Algorithms")

algo_selection = {}
for algo_class in ALGORITHMS:
    if algo_class == "Data Loader":
        st.write(f"#### {algo_class}")
        dataloader = st.radio(
            "Data Loader",
            options=list(ALGORITHMS[algo_class].keys()),
            label_visibility="collapsed",
        )
    else:
        st.write(f"#### {algo_class}")
        algo_selection[algo_class] = {}
        for algo_name in ALGORITHMS[algo_class]:
            algo_selection[algo_class][algo_name] = st.checkbox(algo_name, value=False)

run_button = st.button("Run", type="primary")
if run_button:
    algo_list = [ALGORITHMS["Data Loader"][dataloader]]
    for algo_class in algo_selection:
        for algo_name in algo_selection[algo_class]:
            if algo_selection[algo_class][algo_name]:
                algo_list.append(ALGORITHMS[algo_class][algo_name])

    repository, results = build_and_evaluate_pipelines(algo_list)
    if results:
        st.write("### Pipeline Graph")
        st.code(repository.build_tree_string())
        st.write("### Results")

        records = [
            {
                "Pipeline Tasks": " → ".join(r.get("tasks", [])),
                "Runtime": r.get("runtime", 0),
                "Train Accuracy": r.get("output", {}).get("train_accuracy", 0),
                "Test Accuracy": r.get("output", {}).get("test_accuracy", 0),
            }
            for r in results
        ]
        result_df = pd.DataFrame(records).sort_values(
            by="Test Accuracy", ascending=False
        )
        st.dataframe(result_df)

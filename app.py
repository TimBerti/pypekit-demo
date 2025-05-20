import streamlit as st
import pandas as pd

from pypekit import Repository, CachedExecutor
from algo_repo import ALGORITHMS


def build_and_evaluate_pipelines(algo_list):
    repository = Repository(set(algo_list))
    try:
        repository.build_tree()
        pipelines = repository.build_pipelines()
    except ValueError:
        st.error("No valid pipeline found.")
        return None
    executor = CachedExecutor(pipelines)
    results = executor.run()
    return repository, results

meltpoolnet_df = ALGORITHMS["Data Loader (select at least one)"]["MeltPoolNet Loader"]().run()
iris_df = ALGORITHMS["Data Loader (select at least one)"]["Iris Loader"]().run()
wine_df = ALGORITHMS["Data Loader (select at least one)"]["Wine Loader"]().run()

st.title("Pipeline Synthesis Demo")

st.write("### Datasets")
tab1, tab2, tab3 = st.tabs(["MeltPoolNet Dataset", "Iris Dataset", "Wine Dataset"])
with tab1:
    st.dataframe(iris_df[:5])
with tab2:
    st.dataframe(wine_df[:5])
with tab3:
    st.dataframe(meltpoolnet_df[:5])

st.write("### Algorithms")

algo_selection = {}
for algo_class in ALGORITHMS:
    st.write(f"#### {algo_class}")
    algo_selection[algo_class] = {}
    for algo_name in ALGORITHMS[algo_class]:
        algo_selection[algo_class][algo_name] = st.checkbox(algo_name, value=False)

run_button = st.button("Run", type="primary")
if run_button:
    algo_list = []
    for algo_class in algo_selection:
        for algo_name in algo_selection[algo_class]:
            if algo_selection[algo_class][algo_name]:
                algo_list.append(ALGORITHMS[algo_class][algo_name])

    repository, results = build_and_evaluate_pipelines(algo_list)
    if results:
        st.write("### Results")

        records = [
            {
                "Pipeline Tasks": " → ".join(r.get("tasks", [])),
                "Runtime (s)": round(r.get("runtime", float("nan")), 4),
                "Accuracy": round(r.get("output", float("nan")), 4),
            }
            for r in results
        ]
        result_df = pd.DataFrame(records).sort_values(
            by="Accuracy", ascending=False
        )
        st.dataframe(result_df)

    st.write("### Pipeline Graph")
    st.code(repository.build_tree_string())
import streamlit as st
import pandas as pd

from pypekit import Repository, CachedExecutor
from algo_repo import ALGORITHMS


def build_and_evaluate_pipelines(algo_list):
    repository = Repository(algo_list)
    pipeline_dict = repository.build_pipelines()
    executor = CachedExecutor(pipeline_dict)
    results = executor.run()
    return results

iris_df = ALGORITHMS["Data Loader"]["Iris Loader"]().run()
wine_df = ALGORITHMS["Data Loader"]["Wine Loader"]().run()

st.title("Pipeline Synthesis Demo")

st.write("### Datasets")
tab1, tab2 = st.tabs(["Iris Dataset", "Wine Dataset"])
with tab1:
    st.dataframe(iris_df[:5])
with tab2:
    st.dataframe(wine_df[:5])

st.write("### Algorithms")

algo_selection = {}
for algo_class in ALGORITHMS:
    if type(ALGORITHMS[algo_class]) == dict:
        st.write(f"#### {algo_class}")
        algo_selection[algo_class] = {}
        for algo_name in ALGORITHMS[algo_class]:
            algo_selection[algo_class][algo_name] = st.checkbox(algo_name, value=True)
    else:
        st.write(f"#### {algo_class}")
        algo_selection[algo_class] = st.checkbox(algo_class, value=True)

run_button = st.button("Run", type="primary")
if run_button:
    
    algo_list = []
    for algo_class in algo_selection:
        if type(ALGORITHMS[algo_class]) == dict:
            for algo_name in algo_selection[algo_class]:
                if algo_selection[algo_class][algo_name]:
                    algo_list.append((algo_name, ALGORITHMS[algo_class][algo_name]()))
        else:
            if algo_selection[algo_class]:
                algo_list.append((algo_class, ALGORITHMS[algo_class]()))

    results = build_and_evaluate_pipelines(algo_list)
    st.write("### Results")

    records = [
        {
            "Pipeline Tasks": " → ".join(r.get("tasks", [])),
            "Accuracy": round(r.get("output", float("nan")), 4),
        }
        for r in results.values()
    ]
    result_df = pd.DataFrame(records).sort_values(
        by="Accuracy", ascending=False
    )
    st.dataframe(result_df)
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from usad_beth.config import FIGURES_DIR, REPORTS_DIR

app = typer.Typer()

#function for plotting event order (by timestamp) in dataset
def plot_events_order(df, sorted_df, output_name="event_order.png"):
    hostnames = df.hostName.unique()    

    sns.set_theme()
    sns.set_context("paper")

    fig, axes = plt.subplots(len(hostnames), 2,
                    figsize=(10, 14), sharex=True, sharey=True)


    for ax_i in range(len(hostnames)):
        
        ax = axes[ax_i][0] if len(hostnames)>1 else axes[0]
        
        ax.set_title(hostnames[ax_i] + " (original order)")
        ax.set_ylabel("Secs from boot")
        if ax_i == len(hostnames)-1:
            ax.set_xlabel("Index of event")
        ax.tick_params(axis="x", rotation=55)
        sns.scatterplot(df[df.hostName == hostnames[ax_i]].timestamp,
                        marker="+",
                        ax = ax)

        ax = axes[ax_i][1] if len(hostnames)>1 else axes[1]
        ax.set_title(hostnames[ax_i]+ " (sorted by timestamp)")
        # ax.set_ylabel("Secs from boot")
        if ax_i == len(hostnames)-1:
            ax.set_xlabel("Index of event")
        ax.tick_params(axis="x", rotation=55)
        sns.scatterplot(sorted_df[sorted_df.hostName == hostnames[ax_i]].timestamp,
                        marker="+",
                        ax = ax)

    fig.suptitle("Event order in dataset", y=0.92)
    plt.subplots_adjust(hspace=0.3)
    fig.align_labels()

    fig.savefig(FIGURES_DIR / output_name, dpi=300)

#function for plotting rocauc scores from experiment results
def plot_rocaucs(df_results, sus_or_evil="sus",output_name=FIGURES_DIR / "rocaucs.png"):
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 6)

    sns.set_theme()
    sns.set_context("paper")
    sns.despine(bottom=True, left=True)

    palette = None
    if sus_or_evil == "sus":
        palette = sns.cubehelix_palette(start=1, rot=-1)

    g = sns.barplot(x = "Algorithm" , y = "mean_rocauc",
                hue = "Sorted_enriched",
                data = df_results[df_results["Sus_or_evil"] == "evil"],
                palette=palette)
    
    g.set(xlabel="Algorithm", ylabel="Mean ROCAUC", yticks=np.arange(0.0,1.1,0.1))

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    fig.savefig(FIGURES_DIR / output_name, dpi=300)

#function for plotting proccessing times
def plot_times(df_results, output_name=FIGURES_DIR / "times.png"):
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 6)

    sns.set_theme()
    sns.set_context("paper")
    sns.despine(bottom=True, left=True)

    g = sns.barplot(x = "Algorithm" , y = "mean_time",            
                data = df_results,
                )
    # g.map(sns.barplot, color=".3")
    g.set(xlabel="Algorithm", ylabel="Mean Time, seconds")

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    fig.savefig(FIGURES_DIR / output_name, dpi=300)

@app.command()
def main():
    df_eval_results = pd.read_csv(REPORTS_DIR / "eval_results_full.csv", index_col=0)
    df_results = df_eval_results[["Algorithm", "Time", "ROCAUC_score",
                 "Sus_or_evil", "Sorted_train", "Enriched_parent_process_name"]]\
        .groupby(["Algorithm", "Sus_or_evil", "Sorted_train", "Enriched_parent_process_name"])\
        .agg(
            mean_time=pd.NamedAgg(column="Time", aggfunc="mean"),
            std_time=pd.NamedAgg(column="Time", aggfunc="std"),
            mean_rocauc=pd.NamedAgg(column="ROCAUC_score", aggfunc="mean"),
            std_rocauc=pd.NamedAgg(column="ROCAUC_score", aggfunc="std"),
            max_rocauc=pd.NamedAgg(column="ROCAUC_score", aggfunc="max"),
            min_rocauc=pd.NamedAgg(column="ROCAUC_score", aggfunc="min"),
        ).reset_index()
    df_results["Time"] = df_results.apply(lambda x: f"{x['mean_time']:.3f}" + '±'+ f"{x['std_time']:.3f}", axis=1)
    df_results["ROCAUC"] = df_results.apply(lambda x: f"{x['mean_rocauc']:.3f}" + '±'+ f"{x['std_rocauc']:.3f}", axis=1)
    df_results["Sorted_enriched"] = df_results.apply(lambda x:  x["Sorted_train"] + "," + x["Enriched_parent_process_name"] , axis=1)

    dict = {"no,no": "not sorted, not enriched",
            "no,yes": "not sorted, enriched",
            "yes,no": "sorted, not enriched",
            "yes,yes": "sorted, enriched"}
    df_results.replace({"Sorted_enriched" : dict}, inplace=True)
    plot_rocaucs(df_results)
    plot_rocaucs(df_results, "evil", FIGURES_DIR / "rocaucs_evil.png")
    plot_times(df_results)

if __name__ == "__main__":
    app()

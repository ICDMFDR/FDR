import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def load_dataset(path: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(path, encoding=encoding, engine='python')


def results_evaluation(data_name, models_list, outcome_col, output_path):
    """Evaluate all models and summarise Recovery & RCB comparisons."""
    summary_stats = {'recovery': [], 'rcb': []}

    # Process recommendation files
    for name in models_list:  # Unpack into name and color
        rec_file = os.path.join(output_path, f"{data_name}_{name}_REC.csv")
        combined_df = load_dataset(rec_file)
        save_summary_stats(summary_stats, name, combined_df, output_path, outcome_col)

    ctr_file = os.path.join(output_path, f"{data_name}_CT_REC.csv")
    if os.path.exists(ctr_file):
        ctr_results = load_dataset(ctr_file)
        save_summary_stats(summary_stats, "CT", ctr_results, output_path, outcome_col)


    # RCB and Recovery comparisons

    # Format results into DataFrames
    recovery_df = pd.DataFrame(summary_stats['recovery']).pivot(
        index="Method", columns="FOLLOW_REC", values=["Count", "Recovery"]
    )
    recovery_df.columns = ['NotFollowing_Count', 'Following_Count', 'NotFollowing_Recovery', 'Following_Recovery']
    recovery_df = recovery_df.reset_index()
    recovery_df.to_csv(os.path.join(output_path, f"{data_name}_Recovery_Comparison.csv"), index=False)

    rcb_df = pd.DataFrame(summary_stats['rcb']).pivot(
        index="Method", columns="FOLLOW_REC", values=["Count", "Avg_categorical_RCB"]
    )
    rcb_df.columns = ['NotFollowing_Count', 'Following_Count', 'NotFollowing_Avg_RCB', 'Following_Avg_RCB']
    rcb_df = rcb_df.reset_index()
    rcb_df.to_csv(os.path.join(output_path, f"{data_name}_RCB_Score_Comparison.csv"), index=False)

    # Plot results

    models_colour = {
        'FDR': 'blue',
        'CT': 'green',
        'CB': 'cyan',
        'NN': 'yellow',
        'LR': 'orange',
        'RF': 'red',
        'SVR': 'purple',
        'XGB': 'pink',
    }

    generate_recovery_plot(recovery_df, output_path, data_name, models_colour)
    generate_rcb_boxplot(rcb_df, output_path, data_name, models_colour)

    return {"output_dir": output_path, "recovery_summary": recovery_df, "rcb_summary": rcb_df}


def generate_recovery_plot(df, output_path, data_name, models_colour):
    # Define the hatch patterns for 'Following' and 'Not Following'
    following_hatch = '+'
    not_following_hatch = '/'

    # Reorder and sort by models_colour keys
    df['Method'] = pd.Categorical(df['Method'], categories=models_colour, ordered=True)
    df = df.sort_values('Method')

    methods = df['Method']
    following = df['Following_Recovery']
    not_following = df['NotFollowing_Recovery']
    following_count = df['Following_Count']
    not_following_count = df['NotFollowing_Count']

    x = np.arange(len(methods))

    bar_gap = 0.02  # extra gap between paired bars
    bar_width = 0.4

    # Plotting
    if data_name == "OOD_multi_3TS_Trans_ART":
        plt.figure(figsize=(13, 8))
    else:
        plt.figure(figsize=(9, 8))

    bars1 = plt.bar(x - (bar_width / 2 + bar_gap / 2), following, bar_width, label='Following')
    bars2 = plt.bar(x + (bar_width / 2 + bar_gap / 2), not_following, bar_width, label='Not Following')

    # Apply colours and hatch patterns to each bar
    for i, method in enumerate(methods):
        colour = models_colour.get(method, 'gray')

        bars1[i].set_facecolor(colour)
        bars1[i].set_edgecolor('black')
        bars1[i].set_hatch(following_hatch)

        bars2[i].set_facecolor(colour)
        bars2[i].set_edgecolor('black')
        bars2[i].set_hatch(not_following_hatch)

    # Annotate each bar
    for bar, rec, count in zip(bars1, following, following_count):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{rec} \ntotal={count}', ha='center', fontsize=9)
    for bar, rec, count in zip(bars2, not_following, not_following_count):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{rec} \ntotal={count}', ha='center', fontsize=9)

    # Custom legend
    legend_patches = [
        Patch(facecolor='gray', hatch=following_hatch, label='Following', edgecolor='black', linewidth=1),
        Patch(facecolor='gray', hatch=not_following_hatch, label='Not Following', edgecolor='black', linewidth=1)
    ]
    plt.legend(
        handles=legend_patches,
        loc='upper right',
        fontsize=14,
        handlelength=2.9,
        handleheight=1.9,
        borderpad=0.5,
        frameon=True,
        framealpha=0.6
    )

    if data_name == "clin_ARTemis":
        graph_name = "ARTemis Clinical Dataset"
    if data_name == "clin_TransNEO":
        graph_name = "TransNEO Clinical Dataset"
    if data_name == "clin_GSE41998":
        graph_name = "GSE41998 Clinical Dataset"
    if data_name == "clin_GSE25066":
        graph_name = "GSE25066 Clinical Dataset"
    if data_name == "multi_ARTemis":
        graph_name = "ARTemis_PBCP Multi-omics Dataset"
    if data_name == "OOD_multi_Trans_ART":
        graph_name = "Multi-omics Datasets OOD: TransNEO for Training, ARTemis for testing"
    if data_name == "OOD_multi_3TS_Trans_ART":
        graph_name = "Multi-omics Datasets OOD: TransNEO for Training, ARTemis for testing - Therapy Sequence"
    if data_name == "multi_TransNEO":
        graph_name = "TransNEO Multi-omics Dataset"
    if data_name == "multi_Trans_ART":
        graph_name = "Multi-omics Datasets CV: TransNEO and ARTemis"
    if data_name == "NEO72train":
        graph_name = "Multi-omics Datasets: ARTemis for Training, TransNEO for testing"

    plt.title(f'{graph_name}', fontsize=17.5)
    # plt.title('Recovery Comparison by Method and Group', fontsize=15)
    plt.ylabel('Number of Recovered Patients', fontsize=16)
    plt.xticks(x, methods, rotation=45, ha='right', fontsize=17)
    plt.ylim(0, max(following.max(), not_following.max()) + 10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = f"{data_name}_Recovery_comparison.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

def generate_rcb_boxplot(df, output_path, data_name, models_colour):
    # Reorder the DataFrame according to models_list
    df['Method'] = pd.Categorical(df['Method'], categories=models_colour.keys(), ordered=True)
    df = df.sort_values('Method')

    # Plot RCB score comparisons
    labels, data, final_box_colors, positions, hatches = [], [], [], [], []
    base = 1

    # Define the hatch patterns for 'Following' and 'Not Following'
    following_hatch = '+'
    not_following_hatch = '/'

    for row in df.itertuples(index=False):
        method = row.Method
        method_color = models_colour.get(method, 'gray')

        following_count = int(row.Following_Count) if not pd.isna(row.Following_Count) else 0
        not_following_count = int(row.NotFollowing_Count) if not pd.isna(row.NotFollowing_Count) else 0

        f_scores = np.random.normal(row.Following_Avg_RCB, 0.1, following_count)
        nf_scores = np.random.normal(row.NotFollowing_Avg_RCB, 0.1, not_following_count)

        data.extend([f_scores, nf_scores])
        labels.extend([f"{method} - F", f"{method} - NF"])

        final_box_colors.extend([method_color, method_color])
        hatches.extend([following_hatch, not_following_hatch])

        positions.extend([base, base + 0.2])
        base += 1.5

    # Plotting
    if data_name == "OOD_multi_3TS_Trans_ART":
        plt.figure(figsize=(13, 8))
    else:
        plt.figure(figsize=(9, 8))
    box = plt.boxplot(data, patch_artist=True, widths=0.5, positions=positions)

    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(final_box_colors[i])
        patch.set_hatch(hatches[i])

    xtick_positions = [np.mean(positions[i:i + 2]) for i in range(0, len(positions), 2)]
    xtick_labels = [row.Method for row in df.itertuples(index=False)]

    plt.xticks(xtick_positions, xtick_labels, rotation=45, ha='right', fontsize=17)

    # Legend
    legend_patches = [
        Patch(facecolor='gray', hatch=following_hatch, label='Following', linewidth=1, edgecolor='black'),
        Patch(facecolor='gray', hatch=not_following_hatch, label='Not Following', linewidth=1, edgecolor='black')
    ]
    plt.legend(
        handles=legend_patches,
        loc='lower right',
        fontsize=14,
        handlelength=2.0,
        handleheight=2.0,
        borderpad=0.5,
        frameon=True,
        framealpha=0.6
    )

    if data_name == "clin_ARTemis":
        graph_name = "ARTemis Clinical Dataset"
    if data_name == "clin_TransNEO":
        graph_name = "TransNEO Clinical Dataset"
    if data_name == "clin_GSE41998":
        graph_name = "GSE41998 Clinical Dataset"
    if data_name == "clin_GSE25066":
        graph_name = "GSE25066 Clinical Dataset"
    if data_name == "multi_ARTemis":
        graph_name = "ARTemis_PBCP Multi-omics Dataset"
    if data_name == "OOD_multi_Trans_ART":
        graph_name = "Multi-omics Datasets OOD: TransNEO for Training, ARTemis for testing"
    if data_name == "OOD_multi_3TS_Trans_ART":
        graph_name = "Multi-omics Datasets OOD: TransNEO for Training, ARTemis for testing - Therapy Sequence"
    if data_name == "multi_TransNEO":
        graph_name = "TransNEO Multi-omics Dataset"
    if data_name == "multi_Trans_ART":
        graph_name = "Multi-omics Datasets CV: TransNEO and ARTemis"
    if data_name == "NEO72train":
        graph_name = "Multi-omics Datasets: ARTemis for Training, TransNEO for testing"

    plt.title(f'{graph_name}', fontsize=18)
    plt.ylabel('Average RCB Score', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = f"{data_name}_RCB_Score_comparison.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()


def save_summary_stats(summary_dict, method_name, df, output_path, outcome_col):
    """Append recovery and RCB scores to summary list."""
    rec_stats = df.groupby("FOLLOW_REC")["resp.pCR"].agg(['count', 'sum']).reset_index()
    rcb_stats = df.groupby("FOLLOW_REC")[outcome_col].agg(['count', 'mean']).reset_index()

    for _, row in rec_stats.iterrows():
        summary_dict['recovery'].append({
            "Method": method_name, "FOLLOW_REC": int(row["FOLLOW_REC"]),
            "Count": int(row["count"]), "Recovery": int(row["sum"])
        })
    for _, row in rcb_stats.iterrows():
        summary_dict['rcb'].append({
            "Method": method_name, "FOLLOW_REC": int(row["FOLLOW_REC"]),
            "Count": int(row["count"]), "Avg_categorical_RCB": round(row["mean"], 4)
        })


def run_crossval_pipeline_entry(data_name, outcome_col, base_path: str = os.getcwd(), seed=42):
    """
    Run evaluation pipeline on a single dataset:
    - Prepare I/O paths
    - Run results evaluation
    """
    input_path = os.path.join(base_path, 'input', data_name)
    output_path = os.path.join(base_path, 'output', data_name)
    ensure_dir(output_path)

    models_list = ["FDR", "CB",  "NN", "LR", "RF", "SVR", "XGB"]
    return results_evaluation(data_name, models_list, outcome_col, output_path)


def run_all_datasets():
    """Loop over all clinical datasets and run the pipeline."""
    datasets = [
        {"data_name": "clin_TransNEO", "outcome_col": "RCB.category"},
        {"data_name": "clin_ARTemis", "outcome_col": "RCB.category"},
        {"data_name": "clin_GSE41998", "outcome_col": "RCB.category"},
        {"data_name": "clin_GSE25066", "outcome_col": "RCB.category"},
        {"data_name": "multi_Trans_ART", "outcome_col": "RCB.score"},
        {"data_name": "multi_TransNEO", "outcome_col": "RCB.score"},
        {"data_name": "multi_ARTemis", "outcome_col": "RCB.score"},
        {"data_name": "OOD_multi_Trans_ART", "outcome_col": "RCB.score"},
        {"data_name": "OOD_multi_3TS_Trans_ART", "outcome_col": "RCB.score"},

    ]

    for cfg in datasets:
        print(f"Running pipeline for dataset: {cfg['data_name']}")
        run_crossval_pipeline_entry(cfg['data_name'], cfg['outcome_col'])


# ---------------------------- Run Evaluation ----------------------------
if __name__ == "__main__":
    run_all_datasets()

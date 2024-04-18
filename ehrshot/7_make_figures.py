import os
import argparse
from typing import List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils import (
    LABELERS, 
    TASK_GROUP_2_LABELER,
    HEAD_2_NAME,
    MODEL_2_NAME, 
    SCORE_MODEL_HEAD_2_COLOR,
    filter_df,
    get_rel_path,
    type_tuple_list,
)
from plot import (
    plot_one_labeler,
    plot_one_task_group,
    plot_one_task_group_box_plot,
    _plot_unified_legend,
)

def plot_all_labelers(df_results: pd.DataFrame, 
                                score: str, 
                                path_to_output_dir: str,
                                model_heads: Optional[List[Tuple[str, str]]] = None,
                                is_x_scale_log: bool = True,
                                is_std_bars: bool = True):
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    labelers: List[str] = df_results[df_results['score'] == score]['labeler'].unique().tolist()
    for idx, labeler in enumerate(labelers):
        sub_tasks: List[str] = df_results[(df_results['score'] == score) & (df_results['labeler'] == labeler)]['sub_task'].unique().tolist()
        plot_one_labeler(df_results, 
                                    axes.flat[idx], 
                                    labeler, 
                                    sub_tasks, 
                                    score,
                                    model_heads=model_heads,
                                    is_x_scale_log=is_x_scale_log,
                                    is_std_bars=False if labeler == 'chexpert' else is_std_bars)

    # Create a unified legend for the entire figure
    _plot_unified_legend(fig, axes)

    # Plot aesthetics
    fig.suptitle(f'{score.upper()} by Task', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(os.path.join(path_to_output_dir, f"tasks_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_all_task_groups(df_results: pd.DataFrame, 
                        score: str, 
                        path_to_output_dir: str,
                        model_heads: Optional[List[Tuple[str, str]]] = None,
                        is_x_scale_log: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELER.keys())

    for idx, task_group in enumerate(task_groups):
        plot_one_task_group(df_results, 
                            axes.flat[idx], 
                            task_group, 
                            score,
                            model_heads=model_heads,
                            is_x_scale_log=is_x_scale_log)
    
    # Create a unified legend for the entire figure
    _plot_unified_legend(fig, axes, ncol=2, fontsize=12)

    # Plot aesthetics
    fig.suptitle(f'{score.upper()} by Task Group', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.25)
    plt.savefig(os.path.join(path_to_output_dir, f"taskgroups_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_all_task_group_box_plots(df_results: pd.DataFrame,
                            score: str, 
                            path_to_output_dir: str,
                            model_heads: Optional[List[Tuple[str, str]]] = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELER.keys())

    for idx, task_group in tqdm(enumerate(task_groups)):
        plot_one_task_group_box_plot(df_results, 
                                    axes.flat[idx], 
                                    task_group, 
                                    score,
                                    model_heads=model_heads)
    
    # Create a unified legend for the entire figure
    df_ = filter_df(df_results, score=score, model_heads=model_heads)
    legend_n_col: int = 2
    handles = [ 
        Patch(
            facecolor=SCORE_MODEL_HEAD_2_COLOR[score][model][head], 
            edgecolor=SCORE_MODEL_HEAD_2_COLOR[score][model][head], 
            label=f'{MODEL_2_NAME[model]}+{HEAD_2_NAME[head]}'
        ) 
        for (model, head) in df_[['model', 'head']].drop_duplicates().itertuples(index=False)
    ]
    fig.legend(handles=handles, loc='lower center', ncol=legend_n_col, fontsize=12)
    
    # Plot aesthetics
    fig.suptitle(f'Few-shot v. Full data {score.upper()} by Task Group', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.25)
    plt.savefig(os.path.join(path_to_output_dir, f"taskgroups_boxplot_{score}.png"), dpi=300)
    plt.close('all')
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Make plots of results")
    parser.add_argument("--path_to_results_dir", default=get_rel_path(__file__, "../assets/outputs/"), type=str, help="Path to directory containing results from 7_eval.py")
    parser.add_argument("--path_to_output_dir", default=get_rel_path(__file__, "../assets/figures/"), type=str, help="Path to directory to save figures")
    parser.add_argument("--model_heads", type=type_tuple_list, default=[], help="Specific (model, head) combinations to plot. Format it as a Python list of tuples of strings, e.g. [('clmbr', 'lr'), ('count', 'gbm')]")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    PATH_TO_RESULTS_DIR: str = args.path_to_results_dir
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    MODEL_HEADS: Optional[List[Tuple[str, str]]] = args.model_heads if len(args.model_heads) > 0 else None
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)
    
    # Load all results from CSVs
    dfs: List[pd.DataFrame] = []
    for idx, labeler in tqdm(enumerate(LABELERS)):
        path_to_csv: str = os.path.join(PATH_TO_RESULTS_DIR, f"{labeler}/all_results.csv")
        if os.path.exists(path_to_csv):
            dfs.append(pd.read_csv(path_to_csv))
        else:
            print(f"Ignoring {labeler} b/c no file found at path `{path_to_csv}`")
    df_results: pd.DataFrame = pd.concat(dfs, ignore_index=True)

    # Plotting individual AUROC/AUPRC plot for each labeling function
    for score in tqdm(df_results['score'].unique(), desc='plot_all_labelers()'):
        if score == 'brier': continue
        plot_all_labelers(df_results, score, PATH_TO_OUTPUT_DIR, 
                                    model_heads=MODEL_HEADS, is_x_scale_log=True, is_std_bars=True)

    # Plotting aggregated auroc and auprc plots by task groups
    for score in tqdm(df_results['score'].unique(), desc='plot_all_task_groups()'):
        if score == 'brier': continue
        plot_all_task_groups(df_results, score, path_to_output_dir=PATH_TO_OUTPUT_DIR, 
                             model_heads=MODEL_HEADS, is_x_scale_log=True)

    # plotting aggregated auroc and auprc box plots by task groups
    for score in tqdm(df_results['score'].unique(), desc='plot_all_task_group_box_plots()'):
        if score == 'brier': continue
        plot_all_task_group_box_plots(df_results, score, path_to_output_dir=PATH_TO_OUTPUT_DIR,
                                      model_heads=MODEL_HEADS)
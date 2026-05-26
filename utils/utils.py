import os
import re
import torch
import random
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    matthews_corrcoef, f1_score, balanced_accuracy_score, precision_score, recall_score
)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


feature_pairs = {
    'shape_index': (0, 5),
    'ddc': (1, 6),
    'electrostatics': (2, 7),
    'charge': (3, 8),
    'hydrophobicity': (4, 9),
    'patch_dist':(10,),
    'SASA': (11,12)
}


class Colour:
    """Colour for printing messages."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PINK = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    
def get_date() -> str:
    """Get current date."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    

def log_info(tag: str, msg: str, colour: str = Colour.GREEN) -> None:
    """Log info message."""
    print(f"{colour}[{get_date()}][{tag}]{Colour.RESET} {msg}")


def make_config(data_dir: str) -> dict:
    """Make config."""
    config = {}
    config['dirs'] = {}
    config['dirs']['data_prepare'] = data_dir
    # config['dirs']['grid'] = os.path.join(config['dirs']['data_prepare'], 'static')
    # config['dirs']['md'] = os.path.join(config['dirs']['data_prepare'], 'docked')
    config['dirs']['grid'] = os.path.join(config['dirs']['data_prepare'], '07-grid')
    config['dirs']['md'] = os.path.join(config['dirs']['data_prepare'], 'md')
    
    
    config['dirs']['tmp'] = os.path.join(config['dirs']['data_prepare'], 'tmp')
    os.makedirs(config['dirs']['tmp'], exist_ok=True)
    config['ppi_const'] = {}
    config['ppi_const']['patch_r'] = 16
    os.environ["TMP"] = config['dirs']['tmp']
    os.environ["TMPDIR"] = config['dirs']['tmp']
    os.environ["TEMP"] = config['dirs']['tmp']
    return config


def set_seed(seed: int) -> None:
    """Set seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

def compute_std_mean(train_list_file: str, val_list_file: str, config: dict, model_dir: str, params: dict) -> (np.ndarray, np.ndarray):
    """Compute mean and std of data"""
    # Stack all grid data
    train_list = [x.strip('\n') for x in open(train_list_file, 'r').readlines()]
    val_list = [x.strip('\n') for x in open(val_list_file, 'r').readlines()]
    grid_native_list = []
    for ppi in train_list:
        grid_path = os.path.join(config['dirs']['grid'], ppi, ppi + '.npy')
        if os.path.exists(grid_path):
            grid_native_list.append(np.load(grid_path, allow_pickle=True))
    all_grid = np.stack(grid_native_list, axis=0) # [num_images, patch_size, patch_size, n_features]
    
    # Compute mean and std of data
    radius = config['ppi_const']['patch_r']
    std_array = np.ones(params['n_features'])
    mean_array = np.zeros(params['n_features'])
    
    for feature in feature_pairs.keys(): # For each pair of features
        pixel_values = []
        for feature_i in feature_pairs[feature]: # For each feature in the pair
            for image_i in range(all_grid.shape[0]): # For each image
                for row_i in range(all_grid.shape[1]): # For each row
                    for column_i in range(all_grid.shape[2]): # For each column
                        x = column_i - radius
                        y = radius - row_i
                        if x ** 2 + y ** 2 < radius ** 2:
                            pixel_values.append(all_grid[image_i][row_i][column_i][feature_i])
        mean_value = np.mean(pixel_values)
        std_value = np.std(pixel_values)
        log_info("Data Loading", f"Feature: {feature} -> Mean: {mean_value:.12f}, Std: {std_value:.12f}")
        
        for feature_i in feature_pairs[feature]:
            mean_array[feature_i] = mean_value
            std_array[feature_i] = std_value
        
    mean_array_path = os.path.join(model_dir, 'mean_array.npy')
    np.save(mean_array_path, mean_array)
    std_array_path = os.path.join(model_dir, 'std_array.npy')
    np.save(std_array_path, std_array)
    return train_list, val_list, mean_array, std_array


def get_background_mask(grid: np.ndarray) -> np.ndarray:
    """Get background mask of image."""
    mask = np.zeros((grid.shape[0], grid.shape[1]))
    radius = grid.shape[0] / 2
    for row_i in range(grid.shape[0]):
        for column_i in range(grid.shape[1]):
            # Check if coordinates are within the radius
            x = column_i - radius
            y = radius - row_i
            if x ** 2 + y ** 2 <= radius ** 2:
                mask[row_i][column_i] = 1
    return mask


def save_model_config(model_dir: str, model_name: str, n_params: int, search_space: dict) -> None:
    """Save model config."""
    with open(os.path.join(model_dir, f'{model_name}_config.txt'), 'w') as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f'n_params: {n_params}\n')
        for key, value in search_space.items():
            f.write(f'{key}: {value}\n')
            
            
def select_one_ppi(args: tuple) -> tuple:
    """Sub task: select positive/negative models"""
    ppi, data_dir, docked_grid_dir = args
    parts = ppi.split("_")
    if len(parts) < 3: # The ppi is not valid
        return (ppi, [], [])
    
    pid, ch1, ch2 = parts
    
    metrics_csv = os.path.join(docked_grid_dir, "labels", f"{ppi}_metrics.csv")
    if not os.path.exists(metrics_csv): # The metrics csv file does not exist
        return (ppi, [], [])
    
    
    pos_models, neg_models = [], []
    try:
        metrics = pd.read_csv(metrics_csv)
        for _, row in metrics.iterrows():
            # model name
            model_name = row.get("model_PPI", None)
            
            if model_name is None:
                continue
            
            # model number
            match = re.search(r"model_(\d+)\.pdb", str(model_name))
            number = match.group(1) if match else None
            if number is None:
                continue
            
            # npy file
            grid_file = os.path.join(docked_grid_dir, "07-grid", ppi, f"{ppi}_{number}", f"{ppi}_{number}.npy")
            if not os.path.exists(grid_file):
                continue
            
            image = np.load(grid_file, allow_pickle=True)
            if not np.isfinite(image).all():
                log_info("Warn", f"{grid_file} contains NaN/Inf, skipping.")
                continue
            
            label = row["label"]
            model_i = f"{ppi}_{number}"
            if int(label) == 1:
                pos_models.append(model_i)
            else:
                neg_models.append(model_i)
            
    except Exception as e:
        log_info("Warn", f"{ppi} processed failed, because {e}", Colour.RED)
        return ppi, [], []

    return (ppi, pos_models, neg_models)


def set_environment():
    """Set environment variables."""
    environment_vars = {
        'MSMS_BIN': '/path/msms/msms.x86_64Linux2.2.6.1',
        'PDB2PQR_BIN': '/path/pdb2pqr/pdb2pqr-linux-bin64-2.1.1/pdb2pqr',
        'APBS_BIN': '/path/apbs3.4.1/APBS-3.4.1.Linux/bin/apbs',
        'MULTIVALUE_BIN': '/path/apbs3.4.1/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue'
    }
    for var_name, var_path in environment_vars.items():
        os.environ[var_name] = var_path
        if os.path.exists(var_path):
            log_info("Environment Log", f"Successfully loaded {var_name}.")
        else:   
            log_info("Environment Log", f"Failed to load {var_name}. File not found. You need to install {var_name}.", Colour.RED)
            

def compute_auc(df):
    """Compute the ROC-AUC"""
    labels = df['label'].values
    scores = df['score'].values
    scores = -scores
    fpr, tpr, _ = roc_curve(labels, scores)
    AUC = auc(fpr, tpr)
    return AUC


def compute_ap(df):
    """Compute the AUPRC"""
    labels = df['label'].values
    scores = df['score'].values
    scores = -scores
    AP = average_precision_score(labels, scores)
    return AP

def find_threshold_one_fold(df, score_name, label_name):
    """Find the optimal threshold for a single fold"""
    if df[label_name].nunique() < 2:
        return 0, 0
    precision, recall, thresholds = precision_recall_curve(df[label_name], df[score_name])
    max_matthews = -1
    optimal_threshold = 0
    labels = df[label_name]
    for thr in thresholds:
        pred_labels = df[score_name].apply(lambda x: int(x>thr))
        matthews = matthews_corrcoef(labels, pred_labels)
        if matthews>max_matthews:
            max_matthews = matthews
            optimal_threshold = thr
    return optimal_threshold, max_matthews


def find_optimal_threshold(df, score_name, label_name, reverse_sign=True):
    """Find the optimal threshold for the entire dataset"""
    df = df.copy()
    if reverse_sign:
        df[score_name] = - df[score_name]

    all_thresholds = []
    all_matthews = []
    shuffled = df.sample(frac=1, random_state=15)
    all_chunks = np.array_split(shuffled, 10)
        
    labels = df[label_name]
    for cv_df in all_chunks:
        if len(cv_df) < 2 or cv_df[label_name].nunique() < 2:
            continue
        cv_optimal, cv_matthews = find_threshold_one_fold(cv_df, score_name, label_name)
        all_thresholds.append(cv_optimal)
        all_matthews.append(cv_matthews)
    
    if len(all_thresholds) == 0:
        optimal_threshold = 0
    else:
        optimal_threshold = np.mean(all_thresholds)

        
    pred_labels = df[score_name].apply(lambda x: int(x > optimal_threshold))
    balanced_accuracy = balanced_accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    
    
    if reverse_sign:
        all_thresholds = [-x for x in all_thresholds]
        optimal_threshold = np.mean(all_thresholds)
        
    while len(all_thresholds) < 10:
        all_thresholds.append(0)
        all_matthews.append(0)

    return all_thresholds, all_matthews, optimal_threshold, [balanced_accuracy, f1, precision, recall]


def compute_success_rate(df, ascending):
    """Compute the success rate"""
    df['PPI'] = df['PPI'].str.lower()
    df['PID'] = df['PPI'].str[:4]
    all_targets = list(df['PID'].unique())
    top1 = top10 = top25 = top50 = top75 = top100 = 0
    for target in all_targets:
        target_df = df[df['PID'] == target].sort_values(by='score', ascending=ascending).reset_index(drop=True)
        labels = list(target_df['label'])
        if 1 in labels[:1]: top1 += 1
        if 1 in labels[:10]: top10 += 1
        if 1 in labels[:25]: top25 += 1
        if 1 in labels[:50]: top50 += 1
        if 1 in labels[:75]: top75 += 1
        if 1 in labels[:100]: top100 += 1
    n = len(all_targets)
    return [int(top1*100/n), int(top10*100/n), int(top25*100/n), int(top50*100/n), int(top75*100/n), int(top100*100/n)]


def compute_all_metrics(df):
    """Compute all metrics."""
    auc = compute_auc(df)
    ap = compute_ap(df)
    all_thresholds, all_matthews, optimal_threshold, metrics = find_optimal_threshold(df, 'score', 'label', reverse_sign=True)  
    return [auc, ap] + metrics


def plot_bar(dataset, pos_count, neg_count, model_list, metrics_data, metrics_names, model_colors, out_dir):
    """Plot Metrics Bar Chart"""
    n_metrics = len(metrics_names)
    n_models = len(model_list)

    bar_width = 0.13
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    for i, model in enumerate(model_list):
        name = model
        vals = metrics_data[name]

        ax.bar(
            x + i * bar_width,
            vals,
            width=bar_width,
            color=model_colors[model],
            label=name,
            edgecolor='white',
            linewidth=0.5
        )

        for idx, v in enumerate(vals):
            ax.text(
                x[idx] + i * bar_width,
                v + 0.012,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=5
            )

    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(metrics_names, fontsize=10)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(f"{dataset} - Metrics(pos={pos_count}, neg={neg_count})", fontsize=14, fontweight="bold")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=n_models,
        frameon=False,
        fontsize=9
    )

    plt.tight_layout()

    save_path = os.path.join(out_dir, f"{dataset}_bar_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    log_info("Plot Log", f"Saved to {save_path}")
    
    

def plot_rank(df, pos_count, neg_count, out_dir, model):
    """Plot Benchmark Score Ranking"""
    df["rank"] = np.arange(1, len(df) + 1)

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    last_pos_rank = pos["rank"].max()

    df_left = df[df["rank"] <= last_pos_rank]
    df_right = df[df["rank"] > last_pos_rank]
    neg_right = df_right[df_right["label"] == 0]

    x_scale = 0.1
    df_left = df_left.copy()
    df_left["x"] = df_left["rank"] * x_scale
    box_x = (last_pos_rank + 3) * x_scale

    plt.figure(figsize=(12, 4), dpi=300)

    for _, row in df_left.iterrows():
        x = row["x"]
        y = row["score"]
        label = row["label"]
        color = "#E63946" if label == 1 else "#868686"
        plt.vlines(x, 0, y, color=color, lw=1.8, alpha=0.8)
        plt.scatter(x, y, color=color, s=35, zorder=3)

    scores = neg_right["score"]
    box = plt.boxplot(
        scores,
        positions=[box_x],
        widths=0.3,
        patch_artist=True
    )
    box["boxes"][0].set_facecolor("#868686")

    x_scatter = np.random.normal(box_x, 0.05, len(scores))
    plt.scatter(
        x_scatter, scores,
        color="#868686", s=25, alpha=0.7
    )
    
    plt.title(f"{model} - Benchmark Score Ranking", fontsize=14, fontweight="bold")

    plt.xlabel("Rank", fontsize=13, fontweight="bold")
    plt.ylabel("Prediction score", fontsize=13, fontweight="bold")
    plt.ylim(-2, 2)
    plt.xlim(0, box_x + 0.3)

    tick_positions = np.arange(1, last_pos_rank + 1, 1) * x_scale
    tick_labels = np.arange(1, last_pos_rank + 1, 1)
    plt.xticks(tick_positions, tick_labels)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        Patch(facecolor='#E63946', label=f'Benchmark-positive (n={pos_count})'),
        Patch(facecolor='#868686', label=f'Benchmark-negative (n={neg_count})')
    ]
    plt.legend(handles=legend_elements, loc="upper left", frameon=False)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{model}_rank_score.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    log_info("Plot Log", f"Saved to {save_path}")
    
    
def plot_topk_hot_map(topk_matrix, model_names, out_dir, dataset):
    """Plot Top-K Success Rate Heatmap"""
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        topk_matrix,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Success Rate'},
        xticklabels=['Top1', 'Top10', 'Top25', 'Top50', 'Top75', 'Top100'],
        yticklabels=model_names,
        ax=ax
    )

    ax.set_title(f'{dataset} - Top-K Success Rate', fontsize=14, fontweight='bold')
    ax.set_xlabel('Top-K')
    ax.set_ylabel('Model')

    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{dataset}_topk_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

    log_info("Plot Log", f"Saved to {out_path}")
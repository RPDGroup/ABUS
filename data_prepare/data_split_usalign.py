import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.cluster import AgglomerativeClustering
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import log_info, Colour


def read_pdb_list(config: dict) -> list:
    """Read PDB file list."""
    df = pd.read_csv(config["file_csv"])
    files = df["pdb_path"].tolist()
    files.sort()
    log_info("USalign Log", f"Found {len(files)} interface PDBs.", Colour.BLUE)
    return files


def generate_pairs(files: list) -> list:
    """Generate upper triangle PDB pairs."""
    pairs = []
    n = len(files)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, files[i], files[j]))
    return pairs


def parse_tm_score(stdout: str, stderr: str) -> float:
    """Parse TM-score from USalign output."""
    score = 0.0
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("TM-score="):
            for token in line.split():
                try:
                    score = float(token.strip(","))
                    return score
                except:
                    continue
    if score == 0.0 and stderr:
        for line in stderr.splitlines():
            line = line.strip()
            if line.startswith("TM-score="):
                for token in line.split():
                    try:
                        score = float(token.strip(","))
                        return score
                    except:
                        continue
    return score


def run_usalign_pair(args) -> tuple:
    """Run USalign for one pair."""
    i, j, pdb1, pdb2, config = args
    try:
        res = subprocess.run(
            [config["usalign_path"], pdb1, pdb2, "-mm", "1", "-ter", "1"],
            capture_output=True,
            text=True,
            timeout=config["tmalign_timeout"]
        )
        score = parse_tm_score(res.stdout, res.stderr)
        return (i, j, score)
    except subprocess.TimeoutExpired:
        print(f"[WARN] USalign timeout ({i},{j})")
        return (i, j, 0.0)
    except Exception as e:
        print(f"[ERROR] USalign failed ({i},{j}) : {e}")
        return (i, j, 0.0)


def compute_tm_matrix(files: list, config: dict) -> np.ndarray:
    """Compute pairwise TM-score matrix."""
    n = len(files)
    tm_matrix = np.zeros((n, n), dtype=float)
    pairs = generate_pairs(files)
    if len(pairs) == 0:
        return tm_matrix
    log_info(
        "USalign Log",
        f"Computing {len(pairs)} structure comparisons with {config['n_workers']} workers.",
        Colour.BLUE
    )
    start_time = time.time()
    tasks = [(i, j, p1, p2, config) for (i, j, p1, p2) in pairs]
    with Pool(processes=config["n_workers"]) as pool:
        for idx, result in enumerate(pool.imap_unordered(run_usalign_pair, tasks, chunksize=10), 1):
            i, j, score = result
            tm_matrix[i, j] = score
            tm_matrix[j, i] = score
            if idx % 50 == 0 or idx == len(pairs):
                elapsed = time.time() - start_time
                print(f"done {idx}/{len(pairs)} pairs  elapsed {elapsed:.1f}s")
    np.fill_diagonal(tm_matrix, 1.0)
    return tm_matrix


def save_tm_matrix(tm_matrix: np.ndarray, config: dict) -> None:
    """Save TM-score matrix."""
    np.save(config["tm_matrix_path"], tm_matrix)
    log_info("USalign Log", "TM-score matrix saved.", Colour.BLUE)


def load_tm_matrix(config: dict) -> np.ndarray:
    """Load TM-score matrix."""
    return np.load(config["tm_matrix_path"])


def cluster_structures(tm_matrix: np.ndarray, config: dict) -> np.ndarray:
    """Cluster structures using TM-score."""
    dist_matrix = 1.0 - tm_matrix
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=config["distance_threshold"],
        affinity="precomputed",
        linkage="average"
    )
    labels = clustering.fit_predict(dist_matrix)
    return labels


def split_train_test(files: list, labels: np.ndarray, config: dict) -> None:
    """Split dataset based on clusters."""
    unique_clusters = np.unique(labels)
    np.random.shuffle(unique_clusters)
    split = int(0.8 * len(unique_clusters))
    train_clusters = unique_clusters[:split]
    test_clusters = unique_clusters[split:]
    train_set = [files[i] for i, c in enumerate(labels) if c in train_clusters]
    test_set = [files[i] for i, c in enumerate(labels) if c in test_clusters]
    log_info(
        "Split Log",
        f"Train set: {len(train_set)}  Test set: {len(test_set)}",
        Colour.BLUE
    )
    os.makedirs(os.path.dirname(config["output_train"]), exist_ok=True)
    with open(config["output_train"], "w") as f:
        f.write("\n".join(train_set))
    with open(config["output_test"], "w") as f:
        f.write("\n".join(test_set))


def run_pipeline(config: dict):
    """Full USalign clustering pipeline."""
    log_info("Pipeline", "Start USalign clustering.", Colour.BLUE)
    files = read_pdb_list(config)
    tm_matrix = compute_tm_matrix(files, config)
    save_tm_matrix(tm_matrix, config)
    tm_matrix = load_tm_matrix(config)
    labels = cluster_structures(tm_matrix, config)
    split_train_test(files, labels, config)
    log_info("Pipeline", "Pipeline finished.", Colour.BLUE)


if __name__ == "__main__":

    config = {
        "usalign_path": "/opt/data/private/develop4abbench/zgs/USalign/USalign",
        "file_csv": "/path/pdb_path.csv",
        "output_train": "/path/train_list.txt",
        "output_test": "/path/test_list.txt",
        "tm_matrix_path": "/path/usalign_matrix.npy",
        "n_workers": 180,
        "tmalign_timeout": 60,
        "distance_threshold": 0.5
    }

    run_pipeline(config)
import csv
import random

import numpy as np
import torch
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


# 0. Data loading
features1 = torch.load("")  # TODO: fill in train features path
features2 = torch.load("")  # TODO: fill in test features path
labels1 = torch.load("")    # TODO: fill in train labels path
labels2 = torch.load("")    # TODO: fill in test labels path
indices_train = torch.load("")  # TODO: fill in train indices path
indices_test = torch.load("")   # TODO: fill in test indices path

features1 = features1.numpy()
features2 = features2.numpy()

y_train = labels1.numpy()
y_test = labels2.numpy()

indices_train = np.array(indices_train)
indices_test = np.array(indices_test)

print("features1 shape:", features1.shape)
print("features2 shape:", features2.shape)
print("indices_train shape:", indices_train.shape)
print("indices_test shape:", indices_test.shape)

# torch tensors for simple feature computation
features1_t = torch.from_numpy(features1).float()  # (N, T, C)
features2_t = torch.from_numpy(features2).float()


# 1. Utilities: segment-level → audio-level
def to_int(x):
    """Convert scalar tensor/array or Python number to int."""
    return int(x.item()) if hasattr(x, "item") else int(x)


def aggregate_to_audio_majority(indices_first_col, seg_preds):
    """
    Aggregate segment-level {0,1} predictions to audio-level by majority vote.

    Returns
    -------
    dict[int, bool]
        audio_id -> True/False (prediction > 0.5).
    """
    audio_bucket = {}
    for idx, pv in zip(indices_first_col, seg_preds):
        idx_i = to_int(idx)
        audio_bucket.setdefault(idx_i, []).append(int(pv))
    audio_pred = {aid: (np.mean(v) > 0.5) for aid, v in audio_bucket.items()}
    return audio_pred


def get_audio_true_labels(indices_first_col, seg_labels):
    """
    Get audio-level true labels.

    Assumes all segments from the same audio share the same label.
    Uses the label of the first occurrence of each audio.
    """
    audio_true = {}
    seen = set()
    for idx, y in zip(indices_first_col, seg_labels):
        idx_i = to_int(idx)
        if idx_i not in seen:
            audio_true[idx_i] = int(y)
            seen.add(idx_i)
    return audio_true


# 2. Mean features
print("Computing mean features (train)...")
X_train_mean = features1_t.mean(dim=1).numpy()  # (N, C)
print("Computing mean features (test)...")
X_test_mean = features2_t.mean(dim=1).numpy()


# 3. Single run with mean features
def run_mean_once(seed):
    """Train and evaluate one run of mean-pooling + Bagging-Tree."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    base = DecisionTreeClassifier(random_state=seed)
    clf = BaggingClassifier(
        estimator=base,
        n_estimators=100,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    clf.fit(X_train_mean, y_train)
    seg_pred = clf.predict(X_test_mean)

    audio_pred = aggregate_to_audio_majority(indices_test[:, 0], seg_pred)
    audio_true = get_audio_true_labels(indices_test[:, 0], y_test)

    common_ids = sorted(set(audio_true.keys()) & set(audio_pred.keys()))
    y_true = np.array([audio_true[i] for i in common_ids], dtype=int)
    y_hat = np.array([int(audio_pred[i]) for i in common_ids], dtype=int)

    acc = (y_true == y_hat).mean()
    f1 = f1_score(y_true, y_hat, average="macro")
    pre = precision_score(y_true, y_hat, average="macro")
    rec = recall_score(y_true, y_hat, average="macro")
    return acc, f1, pre, rec


# 4. Multi-seed experiments and CSV saving
NUM_SEEDS = 5
results = []

for s in range(NUM_SEEDS):
    print(f"\n===== Mean method run {s+1}/{NUM_SEEDS} (seed={s}) =====")
    acc, f1, pre, rec = run_mean_once(s)
    print(
        f"[Mean][audio-level] acc={acc:.4f}, f1={f1:.4f}, "
        f"precision={pre:.4f}, recall={rec:.4f}"
    )
    results.append([s, acc, f1, pre, rec])

csv_path = (
    "/home/sichengyu/text/NCDE/SimplifiedProgram/solution/"
    "wav2vec2/Pitt/Pitt_ave_result.csv"
)
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "acc", "f1", "precision", "recall"])
    writer.writerows(results)

print(f"\nAll {NUM_SEEDS} mean runs finished. Results saved to {csv_path}")

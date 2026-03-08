import os
import sys
import gc
import time
import math
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# Pre-allocate GPU memory for JAX
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

sys.path.append('/path/to/log-neural-cdes')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, accuracy_score

import torch
import torch.nn as nn
import torchaudio
import librosa
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import jax.scipy as jsp
import equinox as eqx
import optax
import diffrax

from signax.signature import signature
from signax.signature_flattened import flatten
from signax.tensor_ops import log
from data_dir.hall_set import HallSet

def remat_conv(conv_layer, x):
    return jax.remat(conv_layer)(x)

class VectorField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_size, out_size, width, depth, *, key, scale=1000):
        mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            final_activation=jax.nn.tanh,
            key=key,
        )

        def init_weight(model):
            is_linear = lambda x: isinstance(x, eqx.nn.Linear)
            get_weights = lambda m: [
                x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)
            ]
            weights = get_weights(model)
            new_weights = [weight / scale for weight in weights]
            new_model = eqx.tree_at(get_weights, model, new_weights)
            get_bias = lambda m: [
                x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)
            ]
            biases = get_bias(model)
            new_bias = [bias / scale for bias in biases]
            new_model = eqx.tree_at(get_bias, new_model, new_bias)
            return new_model

        self.mlp = init_weight(mlp)

    def __call__(self, y):
        return self.mlp(y)


class NeuralCDE(eqx.Module):
    vf: eqx.nn.MLP 
    data_dim: int 
    hidden_dim: int 
    ode_solver_stepsize: int 
    linear1: eqx.nn.Linear 
    linear2: eqx.nn.Linear 

    conv1: eqx.nn.Conv
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    dropout3: eqx.nn.Dropout
    ln1: eqx.nn.LayerNorm

    def __init__(
        self,
        hidden_dim,
        data_dim,
        label_dim,
        vf_hidden_dim,
        vf_num_hidden,
        ode_solver_stepsize,
        *,
        key,
    ):
        vf_key, l1key, l2key, conv_key = jr.split(key, 4)
                
        self.vf = VectorField(
            hidden_dim,
            hidden_dim * data_dim,
            vf_hidden_dim,
            vf_num_hidden,
            scale=1,
            key=vf_key,
        )
        self.linear1 = eqx.nn.Linear(data_dim, hidden_dim, key=l1key)
        self.linear2 = eqx.nn.Linear(hidden_dim, label_dim, key=l2key)
        
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.ode_solver_stepsize = ode_solver_stepsize
        
        self.conv1 = eqx.nn.Conv(
            1,
            in_channels=768,
            out_channels=8,
            kernel_size=1,
            key=conv_key
        )

        self.ln1 = eqx.nn.LayerNorm(shape=10, eps=1e-8, use_weight=False, use_bias=False)
        self.dropout1 = eqx.nn.Dropout(0.4) 
        self.dropout2 = eqx.nn.Dropout(0.5)
        self.dropout3 = eqx.nn.Dropout(0.2)

    def get_ode(self, ts, X):
        coeffs = diffrax.backward_hermite_coefficients(ts, X)
        control = diffrax.CubicInterpolation(ts, coeffs)
        func = lambda t, y, args: jnp.reshape(
            self.vf(y), (self.hidden_dim, self.data_dim)
        )
        return diffrax.ControlTerm(func, control).to_ode(), control
    
    def __call__(self, X, key, *, inference=False):
        key, dropout_key1, dropout_key2, dropout_key3 = jr.split(key, 4)
        ts = X[:, 0]
        X_features = X[:, 1:]
        X_with_time = jnp.concatenate([ts[:, None], X_features], axis=1)
        
        result = self.get_ode(ts, X_with_time)
    
        if isinstance(result, tuple):
            ode_term, control = result
            h0 = self.linear1(control.evaluate(ts[0]))
        else:
            ode_term = result
            h0 = self.linear1(X_with_time[0, :])
    
        saveat = diffrax.SaveAt(t1=True)
        solution = jax.remat(diffrax.diffeqsolve(
            terms=ode_term,
            solver=diffrax.Heun(),
            t0=ts[0],
            t1=ts[-1],
            dt0=self.ode_solver_stepsize,
            y0=h0,
            saveat=saveat,
            stepsize_controller=diffrax.ConstantStepSize(),
        ))
        
        (prediction,) = jnn.sigmoid(self.linear2(solution.ys[-1]))
        return prediction

class LogNeuralCDE(NeuralCDE):
    stepsize: int 
    depth: int
    hall_set: HallSet

    def __init__(
        self,
        hidden_dim,
        data_dim,
        label_dim,
        vf_hidden_dim,
        vf_num_hidden,
        ode_solver_stepsize,
        stepsize,
        depth,
        *,
        key,
    ):
        super().__init__(
            hidden_dim,
            data_dim,
            label_dim,
            vf_hidden_dim,
            vf_num_hidden,
            ode_solver_stepsize,
            key=key,
        )
        self.stepsize = stepsize
        if depth not in [1, 2]:
            raise ValueError("Log-ODE method is only implemented for truncation depths one and two")
        self.depth = depth
        self.hall_set = HallSet(data_dim, depth)

    def calc_logsigs(self, X):
        X = X.reshape(-1, self.stepsize, X.shape[-1])
        prepend = jnp.concatenate((jnp.zeros((1, X.shape[-1])), X[:-1, -1, :]))[:, None, :]
        X = jnp.concatenate((prepend, X), axis=1)

        def logsig(x):
            logsig = flatten(log(signature(x, self.depth)))
            if self.depth == 1:
                return jnp.concatenate((jnp.array([0]), logsig))
            else:
                tensor_to_lie_map = self.hall_set.t2l_matrix(self.depth)
                return tensor_to_lie_map[:, 1:] @ logsig

        logsigs = jax.vmap(logsig)(X)
        return logsigs

    def depth_one_ode(self, y, logsig, interval_length):
        vf_out = jnp.reshape(self.vf(y), (self.hidden_dim, self.data_dim))
        return jnp.dot(vf_out, logsig[1:]) / interval_length

    def depth_two_ode(self, y, logsig, interval_length):
        vf_out = jnp.reshape(self.vf(y), (self.hidden_dim, self.data_dim))

        jvps = jnp.reshape(
            jax.vmap(lambda x: jax.jvp(self.vf, (y,), (x,))[1])(vf_out.T),
            (self.data_dim, self.data_dim, self.hidden_dim),
        )

        def liebracket(jvps, pair):
            return jvps[pair[0] - 1, pair[1] - 1] - jvps[pair[1] - 1, pair[0] - 1]

        pairs = jnp.asarray(self.hall_set.data[self.data_dim + 1 :])
        lieout = jax.vmap(liebracket, in_axes=(None, 0))(jvps, pairs)

        vf_depth1 = jnp.dot(vf_out, logsig[1 : self.data_dim + 1])
        vf_depth2 = jnp.dot(lieout.T, logsig[self.data_dim + 1 :])

        return (vf_depth1 + vf_depth2) / interval_length

    def get_ode(self, ts, X):
        logsigs = self.calc_logsigs(X)
        intervals = (jnp.arange(0, X.shape[0] + self.stepsize, self.stepsize) / X.shape[0])

        def func(t, y, args):
            idx = jnp.searchsorted(intervals, t)
            logsig_t = logsigs[idx - 1]
            interval_length = intervals[idx] - intervals[idx - 1]
            if self.depth == 1:
                return self.depth_one_ode(y, logsig_t, interval_length)
            if self.depth == 2:
                return self.depth_two_ode(y, logsig_t, interval_length)

        return diffrax.ODETerm(func)

i = 12
last = 1
single = 12

features1 = np.load(f'/path/to/features/train_features_NCMMSC2021_single{single}.npy')
features2 = np.load(f'/path/to/features/test_features_NCMMSC2021_single{single}.npy')
labels1 = torch.load(f'/path/to/labels/labels1_whisper_single{single}_0.3_30s_train.pt')
labels2 = torch.load(f'/path/to/labels/labels2_whisper_single{single}_0.3_30s_test.pt')
indices_train = torch.load(f'/path/to/labels/indices_whisper_single{single}_0.3_30s_train.pt')
indices_test = torch.load(f'/path/to/labels/indices_whisper_single{single}_0.3_30s_test.pt')

features_np = features1
features_np_test = features2
labels1_np = labels1.detach().cpu().numpy()
labels2_np = labels2.detach().cpu().numpy()

features_jax = jnp.array(features_np)
features_jax_test = jnp.array(features_np_test)
labels_jax = jnp.array(labels1_np)
labels_jax_test = jnp.array(labels2_np)

def preprocess_data(features):
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    standardized_features = (features - mean) / (std + 1e-8)
    return standardized_features

def get_data(features):
    ts = jnp.linspace(0, 1, features.shape[1])
    ts1 = jnp.repeat(ts[None, :], features.shape[0], axis=0)
    normalized_features = preprocess_data(features)
    time_steps_expanded = ts1[:, :, None]
    features_with_time = jnp.concatenate([time_steps_expanded, normalized_features], axis=2)
    return features_with_time

X_train = get_data(features_jax)
X_test = get_data(features_jax_test)

y_train = labels_jax
y_test = labels_jax_test

def count_audio_files(directory):
    audio_extensions = ('.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma')
    audio_file_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_file_count += 1
    return audio_file_count

train_ccn = count_audio_files("/path/to/dataset/train/HC")
train_cdn = count_audio_files("/path/to/dataset/train/AD")
test_ccn = count_audio_files("/path/to/dataset/test/HC")
test_cdn = count_audio_files("/path/to/dataset/test/AD")

print("train_ccn:", train_ccn)
print("train_cdn:", train_cdn)
print("test_ccn:", test_ccn)
print("test_cdn:", test_cdn)

labels_test = jnp.concatenate([jnp.zeros(test_ccn), jnp.ones(test_cdn)])
labels_train = jnp.concatenate([jnp.zeros(train_ccn), jnp.ones(train_cdn)])

class Dataloader:
    data: jnp.ndarray
    labels: jnp.ndarray
    size: int

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.size = len(data)

    def loop(self, batch_size, *, key=None):
        if batch_size == self.size:
            yield self.data, self.labels

        indices = jnp.arange(self.size)
        while True:
            subkey, key = jr.split(key)
            perm = jr.permutation(subkey, indices)
            start = 0
            end = batch_size
            while end < self.size:
                batch_perm = perm[start:end]
                yield self.data[batch_perm], self.labels[batch_perm]
                start = end
                end = start + batch_size

train_dataloader = Dataloader(X_train, y_train)
test_dataloader = Dataloader(X_test, y_test)

@eqx.filter_jit
@eqx.filter_value_and_grad
def classification_loss(model, X, y, *, key):
    batch_size = X.shape[0]
    keys = jax.random.split(key, batch_size)

    def model_forward(x, k):
        return model(x, k, inference=False)

    pred_y = jax.vmap(model_forward)(X, keys)
    epsilon = 1e-7

    pred_y_clipped = jnp.clip(pred_y, epsilon, 1 - epsilon)
    loss = - (y * jnp.log(pred_y_clipped) + (1 - y) * jnp.log(1 - pred_y_clipped))
    norm = 0
    for layer in model.vf.mlp.layers:
        norm += jnp.mean(
            jnp.linalg.norm(layer.weight, axis=-1)
            + jnp.linalg.norm(layer.bias, axis=-1)
        )
    norm *= 0
    return jnp.mean(loss) + norm

@eqx.filter_jit
def train_step(model, X, y, opt, opt_state, *, key):
    key, subkey = jr.split(key)
    loss, grads = classification_loss(model, X, y, key=subkey)
    updates, opt_state = opt.update(grads, opt_state, params=trainable_params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def save_to_excel(new_hyperparameters, file_name='hyperparameters_results.xlsx'):
    directory = os.path.dirname(file_name)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    try:
        existing_df = pd.read_excel(file_name)
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading file: {e}. Creating a new file.")
        existing_df = pd.DataFrame()
    
    new_df = pd.DataFrame(new_hyperparameters)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_excel(file_name, index=False)

def append_and_save_hyperparameters(length, cover, batch_size, lr, total_steps, hidden_size, width_size, depth, dropout_rate, seed, accuracy, accuracy2, precision, recall, f1, file_name):
    current_hyperparameters = [{
        'length': length,
        'cover': cover,
        'batch_size': batch_size,
        'lr': lr,
        'total_steps': total_steps,
        'hidden_size': hidden_size,
        'width_size': width_size,
        'depth': depth,
        'dropout_rate': dropout_rate,
        'seed': seed,
        'acc_test': accuracy,
        'acc_test_vote': accuracy2,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }]
    save_to_excel(current_hyperparameters, file_name)


def evaluate_and_plot_roc_pr_curves(labels, predictions, plot_title_prefix=""):
    labels_np = jnp.asarray(labels)
    predictions_np = jnp.asarray(predictions)

    if labels_np.shape[0] != predictions_np.shape[0]:
        raise ValueError(f"Inconsistent number of samples: {labels_np.shape[0]}, {predictions_np.shape[0]}")

    fpr, tpr, roc_thresholds = roc_curve(labels_np, predictions_np)
    roc_auc = auc(fpr, tpr)
    print(f"{plot_title_prefix} AUC (ROC):", roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC (area = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{plot_title_prefix} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    youden_j = tpr - fpr
    best_threshold_index_roc = jnp.argmax(youden_j)
    best_threshold_roc = roc_thresholds[best_threshold_index_roc]

    precision, recall, pr_thresholds = precision_recall_curve(labels_np, predictions_np)
    pr_thresholds = pr_thresholds[:-1]
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = jnp.nan_to_num(f1_scores, nan=0.0)

    best_threshold_index_pr = jnp.argmax(f1_scores)
    if best_threshold_index_pr >= len(pr_thresholds):
        best_threshold_index_pr = len(pr_thresholds) - 1
    best_threshold_pr = pr_thresholds[best_threshold_index_pr]

    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'PR (F1 max = {f1_scores[best_threshold_index_pr]:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{plot_title_prefix} Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    accuracies = []
    for threshold in pr_thresholds:
        preds = (predictions_np >= threshold).astype(int)
        accuracy = accuracy_score(labels_np, preds)
        accuracies.append(accuracy)
    accuracies = jnp.array(accuracies)

    best_threshold_index_accuracy = jnp.argmax(accuracies)
    best_threshold_accuracy = pr_thresholds[best_threshold_index_accuracy]

    plt.figure()
    plt.plot(pr_thresholds, accuracies, color='r', lw=2, label=f'Accuracy (max = {accuracies[best_threshold_index_accuracy]:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title(f'{plot_title_prefix} Accuracy vs. Threshold')
    plt.legend(loc="lower left")
    plt.show()

    return best_threshold_roc, best_threshold_pr, best_threshold_accuracy

def save_model_results(
    seed,
    acc_test,
    f1,
    precision,
    recall,
    acc_test_vote,
    f1_vote,
    precision_vote,
    recall_vote,
    test_predictions1,
    test_predictions2,
    metrics_csv="metrics.csv",
    preds_csv="predictions.csv"
):
    if not os.path.exists(metrics_csv):
        df_metrics = pd.DataFrame(columns=["seed","acc_test","f1","precision","recall","acc_test_vote","f1_vote","precision_vote","recall_vote"])
    else:
        df_metrics = pd.read_csv(metrics_csv)

    if seed in df_metrics["seed"].values:
        print(f"[INFO] Metrics for Seed={seed} already exist in {metrics_csv}, skipping.")
    else:
        new_row_df = pd.DataFrame([{
            "seed": seed,
            "acc_test": acc_test,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "acc_test_vote": acc_test_vote,
            "f1_vote": f1_vote,
            "precision_vote": precision_vote,
            "recall_vote": recall_vote
        }])
        df_metrics = pd.concat([df_metrics, new_row_df], ignore_index=True)
        df_metrics.to_csv(metrics_csv, index=False)
        print(f"[SUCCESS] Metrics for Seed={seed} saved to {metrics_csv}.")

    if not os.path.exists(preds_csv):
        df_preds = pd.DataFrame(columns=["seed","test_predictions1","test_predictions2"])
    else:
        df_preds = pd.read_csv(preds_csv)

    if seed in df_preds["seed"].values:
        print(f"[INFO] Predictions for Seed={seed} already exist in {preds_csv}, skipping.")
    else:
        test_preds1_str = str(test_predictions1)
        test_preds2_str = str(test_predictions2)

        new_row_preds_df = pd.DataFrame([{
            "seed": seed,
            "test_predictions1": test_preds1_str,
            "test_predictions2": test_preds2_str
        }])
        df_preds = pd.concat([df_preds, new_row_preds_df], ignore_index=True)
        df_preds.to_csv(preds_csv, index=False)
        print(f"[SUCCESS] Predictions for Seed={seed} saved to {preds_csv}.")

def get_trainable_params(model):
    return eqx.filter(model, eqx.is_inexact_array)

def train_model(
    model,
    num_steps=240, 
    print_steps=24, 
    batch_size=32, 
    base_lr=3.5e-4, 
    warmup_steps = 48, 
    weight_decay=0, 
    *,
    key,
    seed,
):
    global train_predictions1, test_predictions1, test_predictions2, trainable_params
    trainable_params = get_trainable_params(model)

    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )
    
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=num_steps - warmup_steps,
        alpha=0.01
    )
    
    lr_schedule = optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule], 
        boundaries=[warmup_steps]
    )
    
    opt = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    test_accs = []
    test_accs_vote = []
    steps = []
    train_accs = []

    dataset_size = X_train.shape[0]
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    total_epochs = math.ceil(num_steps / steps_per_epoch)

    for epoch in range(total_epochs):
        print(f"Epoch: {epoch + 1}")
        trainloopkey, key = jax.random.split(key)
        
        for step, data in zip(
            range(steps_per_epoch), train_dataloader.loop(batch_size, key=trainloopkey)
        ):
            start_time = time.time()
    
            X, y = data
            key, subkey = jr.split(key)
            
            model, opt_state, loss = train_step(model, X, y, opt, opt_state, key=subkey)
            
            if step == 0 or (step + 1) % print_steps == 0 or step == (steps_per_epoch - 1):
                inference_model = eqx.nn.inference_mode(model)
                inference_model = eqx.Partial(inference_model, inference=True)
                
                for batch, data in zip(range(1), train_dataloader.loop(train_dataloader.size)):
                    X, y = data
                    keys = jax.random.split(jr.PRNGKey(0), X.shape[0])
                    output = jax.vmap(inference_model)(X, keys)
                    pre_train = output
                    train_acc = jnp.mean((output > 0.5) == (y == 1))

                for batch, data in zip(range(1), test_dataloader.loop(test_dataloader.size)):
                    X, y = data
                    keys = jax.random.split(jr.PRNGKey(0), X.shape[0])
                    output = jax.vmap(inference_model)(X, keys)
                    test_acc = jnp.mean((output > 0.5) == (y == 1))
                    
                if step == steps_per_epoch - 1:
                    pre_test = output

                elapsed_time = time.time() - start_time
                print(f"Step: {step + 1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {elapsed_time:.4f} seconds")
                steps.append(step + 1)
                
        audio_segments_train = {}       
        for idx, pred in zip(indices_train[:,0], pre_train):
            if idx.size == 1:
                idx = int(idx.item())
            else:
                raise ValueError(f"Unexpected idx size: {idx.size}, idx: {idx}")
            if idx in audio_segments_train:
                audio_segments_train[idx].append(pred)
            else:
                audio_segments_train[idx] = [pred]
        
        audio_predictions_train = {idx: jnp.mean(jnp.array(preds)) for idx, preds in audio_segments_train.items()}
        predictions1 = list(audio_predictions_train.values())
        predictions1 = jnp.array(predictions1)

        train_predictions1 = [(idx, 1 if pred >= 0.5 else 0) for idx, pred in audio_predictions_train.items()]
                    
        audio_segments_test = {}
        for idx, pred_val in zip(indices_test[:, 0], pre_test):
            if idx.size == 1:
                idx = int(idx.item()) 
            else:
                raise ValueError(f"Unexpected idx size: {idx.size}, idx: {idx}")
            if idx in audio_segments_test:
                audio_segments_test[idx].append(pred_val)
            else:
                audio_segments_test[idx] = [pred_val]
        
        audio_predictions_test = {idx: jnp.mean(jnp.array(preds)) for idx, preds in audio_segments_test.items()}
        audio_predictions_test_vote = {idx: 1 if jnp.sum(jnp.array(preds) > 0.5) > jnp.sum(jnp.array(preds) <= 0.5) else 0 for idx, preds in audio_segments_test.items()}
        
        values = jnp.array(list(audio_predictions_test.values()))
        mean_value_test = jnp.mean(values)
        mean_value_train = jnp.mean(pre_train)
        
        print("mean_value_train", mean_value_train)
        print("mean_value_test", mean_value_test)

        correct_predictions_test = 0
        predict_label = []
        predict_label_vote = []
        for idx, pred in audio_predictions_test.items():
            label = labels_test[idx]
            predict_label.append((pred > 0.5))
            if (pred > 0.5) == label:
                correct_predictions_test += 1
                
        acc_test = correct_predictions_test / len(audio_predictions_test)
                
        correct_predictions_test_vote = 0
        for idx, pred in audio_predictions_test_vote.items():
            label = labels_test[idx]
            predict_label_vote.append(pred)
            if pred == label:
                correct_predictions_test_vote += 1

        acc_test_vote = correct_predictions_test_vote / len(audio_predictions_test)
        
        predict_label_array = np.array(predict_label)
        predict_label_vote_array = np.array(predict_label_vote)
        labels_test_array = np.array(labels_test)
        
        precision = precision_score(labels_test_array, predict_label_array)
        precision_vote = precision_score(labels_test_array, predict_label_vote_array)
        recall = recall_score(labels_test_array, predict_label_array)
        recall_vote = recall_score(labels_test_array, predict_label_vote_array)
        f1 = f1_score(labels_test_array, predict_label_array)
        f1_vote = f1_score(labels_test_array, predict_label_vote_array)

        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'acc_test: {acc_test:.4f}')
        print(f'acc_test_vote: {acc_test_vote:.4f}')
        
        test_accs.append(acc_test)
        test_accs_vote.append(acc_test_vote)
        
        if epoch == total_epochs - 1:
            predictions_list_test = []
            for idx, preds in audio_segments_test.items():
                mean_pred = jnp.mean(jnp.array(preds))
                predictions_list_test.append((idx, mean_pred))
            
            predictions_list_test.sort(key=lambda x: x[0])
            
            for idx, mean_pred in predictions_list_test:
                print(f"Audio segment test {idx}: Prediction value {mean_pred:.4f}")
            test_predictions1 = [(idx, 1 if pred >= 0.5 else 0) for idx, pred in predictions_list_test]
            test_predictions2 = [(idx, 1 if pred == 1 else 0) for idx, pred in audio_predictions_test_vote.items()]
            save_model_results(
                seed=seed,
                acc_test=acc_test,
                f1=f1,
                precision=precision,
                recall=recall,
                acc_test_vote=acc_test_vote,
                f1_vote=f1_vote,
                precision_vote=precision_vote,
                recall_vote=recall_vote,
                test_predictions1=test_predictions1,
                test_predictions2=test_predictions2,
                metrics_csv=f"/path/to/solution/whisper_log_single{single}_auto(1)_(1).csv",
                preds_csv=f"/path/to/solution/whisepr_log_single{single}_predict_auto(1)_(1).csv"
            )
        
    return acc_test, acc_test_vote, test_accs, test_accs_vote, train_predictions1, test_predictions1, test_predictions2

class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_val_acc = 0
        self.patience_counter = 0
        self.best_metrics = {}
        self.best_epoch = 0

    def __call__(self, model, val_acc, epoch, metrics):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch + 1
            self.best_metrics = metrics
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                return True 
        return False

def train_with_seeds(seeds):
    global hyperparameters, all_train_predictions, all_test_predictions_vote, all_test_predictions, acc_seeds
    hyperparameters = []
    all_train_predictions = [] 
    all_test_predictions = [] 
    all_test_predictions_vote = [] 
    acc_seeds = [] 
    all_test_accuracies = []
    all_test_accuracies_vote = []
    
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        modelkey, key = jax.random.split(key)
        trainkey, key = jax.random.split(key)

        LogNCDE_Depth2 = LogNeuralCDE(
            hidden_dim=hidden_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            vf_hidden_dim=vf_hidden_dim,
            vf_num_hidden=vf_num_hidden,
            ode_solver_stepsize=ode_solver_stepsize,
            stepsize=stepsize,
            depth=2,
            key=modelkey,
        )
        
        try:
            print(f"Training with seed: {seed}")
            train_predictions1 = []
            test_predictions1 = []
            test_predictions2 = []

            acc_test, acc_test_vote, test_accs, test_accs_vote, train_predictions1, test_predictions1, test_predictions2 = train_model(LogNCDE_Depth2, key=trainkey, seed=seed)

            acc_test_cpu = np.array(acc_test)
            acc_test_vote_cpu = np.array(acc_test_vote)
            test_accs_cpu = np.array(test_accs)
            test_accs_vote_cpu = np.array(test_accs_vote)

            test_predictions1_cpu = [np.array(p) for p in test_predictions1]
            test_predictions2_cpu = [np.array(p) for p in test_predictions2]

            all_train_predictions.append(train_predictions1)
            all_test_predictions.append(test_predictions1_cpu)
            all_test_predictions_vote.append(test_predictions2_cpu)
            all_test_accuracies.append(test_accs_cpu)
            all_test_accuracies_vote.append(test_accs_vote_cpu)

        except Exception as e:
            print(f"Error encountered with seed {seed}: {e}")
            placeholder_accs = np.zeros(len(test_accs)) if 'test_accs' in locals() else np.zeros(10)
            all_test_accuracies.append(placeholder_accs)
            placeholder_accs_vote = np.zeros(len(test_accs_vote)) if 'test_accs_vote' in locals() else np.zeros(10)
            all_test_accuracies_vote.append(placeholder_accs_vote)
            continue

        finally:
            del LogNCDE_Depth2
            del key
            del modelkey
            del trainkey
            jax.device_put(None)
            gc.collect()
            jax.clear_caches()

    print("test_predictions:", all_test_predictions)
    print("test_predictions_vote:", all_test_predictions_vote)

    vote_and_evaluate()

    epochs = range(1, len(all_test_accuracies[0]) + 1)
    df_test = pd.DataFrame(all_test_accuracies, columns=epochs, index=[f"Seed_{s}" for s in seeds])
    df_test_vote = pd.DataFrame(all_test_accuracies_vote, columns=epochs, index=[f"Seed_{s}" for s in seeds])
    print("\nTest Accuracies per Epoch per Seed:")
    print(df_test)
    print("\nTest Accuracies vote per Epoch per Seed:")
    print(df_test_vote)
    return df_test

def vote_and_evaluate():
    train_votes = aggregate_votes(all_train_predictions)
    accuracy = evaluate_accuracy(train_votes, labels_train, "Train")

    test_votes = aggregate_votes(all_test_predictions)
    accuracy_test = evaluate_accuracy(test_votes, labels_test, "Test")
    test_votes = jnp.array(list(test_votes.values()))

    test_vote_votes = aggregate_votes(all_test_predictions_vote)
    accuracy_vote_test = evaluate_accuracy(test_vote_votes, labels_test, "Test_vote")
    test_vote_votes = jnp.array(list(test_vote_votes.values()))
    
    precision = precision_score(labels_test, test_votes)
    print(f'Precision: {precision:.4f}')
    
    recall = recall_score(labels_test, test_votes)
    print(f'Recall: {recall:.4f}')
    
    f1 = f1_score(labels_test, test_votes)
    print(f'F1 Score: {f1:.4f}')

    precision_vote = precision_score(labels_test, test_vote_votes)
    print(f'Precision_vote: {precision_vote:.4f}')
    
    recall_vote = recall_score(labels_test, test_vote_votes)
    print(f'Recall_vote: {recall_vote:.4f}')
    
    f1_vote = f1_score(labels_test, test_vote_votes)
    print(f'F1 Score_vote: {f1_vote:.4f}')
    
    append_and_save_hyperparameters(12800, 0.3, 64, 3e-4, 128, 8, 64, 2, 0, 1, accuracy_test, accuracy_vote_test, precision_vote, recall_vote, f1_vote, file_name='/path/to/solution/hyperparameters_results_2021_majvote.xlsx')
    print('testvotes:', test_votes)
    print('test_vote_votes:', test_vote_votes)

def aggregate_votes(predictions_list_all_seeds):
    aggregated_predictions = {}
    for idx in range(len(predictions_list_all_seeds[0])):
        preds = [predictions_list_all_seeds[seed_idx][idx][1] for seed_idx in range(len(predictions_list_all_seeds))]
        vote_result = 1 if sum(pred == 1 for pred in preds) > len(preds) / 2 else 0
        aggregated_predictions[idx] = vote_result
    return aggregated_predictions

def evaluate_accuracy(aggregated_predictions, labels, dataset_name):
    correct_predictions = 0
    for idx, pred in aggregated_predictions.items():
        label = labels[idx]
        if pred == label:
            correct_predictions += 1
    accuracy = correct_predictions / len(aggregated_predictions)
    print(f'{dataset_name} Accuracy (after voting): {accuracy:.4f}')
    return accuracy

if __name__ == "__main__":
    hidden_dim = 128
    data_dim = 65
    label_dim = 1
    vf_hidden_dim = 512
    vf_num_hidden = 3
    ode_solver_stepsize = 1 / 250
    stepsize = 60
    num_seeds = 5 
    seeds = np.arange(1001, 1006).tolist()
    test_seeds = [1001, 1002, 1003, 1004, 1005]
    print("seed:", test_seeds)
    train_with_seeds(seeds)

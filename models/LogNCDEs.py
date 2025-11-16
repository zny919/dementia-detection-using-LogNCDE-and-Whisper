import os
import math
import time
import gc
import warnings
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
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
                x.weight
                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                if is_linear(x)
            ]
            weights = get_weights(model)
            new_weights = [weight / scale for weight in weights]
            new_model = eqx.tree_at(get_weights, model, new_weights)

            get_bias = lambda m: [
                x.bias
                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                if is_linear(x)
            ]
            biases = get_bias(model)
            new_bias = [bias / scale for bias in biases]
            new_model = eqx.tree_at(get_bias, new_model, new_bias)
            return new_model

        self.mlp = init_weight(mlp)

    def __call__(self, y):
        return self.mlp(y)


class NeuralCDE(eqx.Module):
    """Standard Neural CDE with an MLP vector field."""

    vf: VectorField
    data_dim: int
    hidden_dim: int
    ode_solver_stepsize: float
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    dropout3: eqx.nn.Dropout
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

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
        vf_key, l1key, l2key, conv_key, conv_key2 = jr.split(key, 5)

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

        # Simple Conv1d front-end (currently commented out in __call__)
        self.conv1 = eqx.nn.Conv(
            1,
            in_channels=768,
            out_channels=8,
            kernel_size=1,
            key=conv_key,
        )
        self.conv2 = eqx.nn.Conv(
            1,
            in_channels=128,
            out_channels=8,
            kernel_size=3,
            padding=1,
            key=conv_key2,
        )

        self.ln1 = eqx.nn.LayerNorm(shape=8, eps=1e-8, use_weight=False, use_bias=False)
        self.ln2 = eqx.nn.LayerNorm(shape=8, eps=1e-8, use_weight=False, use_bias=False)

        self.dropout1 = eqx.nn.Dropout(0.5)
        self.dropout2 = eqx.nn.Dropout(0.0)
        self.dropout3 = eqx.nn.Dropout(0.5)

    def get_ode(self, ts, X):
        coeffs = diffrax.backward_hermite_coefficients(ts, X)
        control = diffrax.CubicInterpolation(ts, coeffs)

        def func(t, y, args):
            return jnp.reshape(self.vf(y), (self.hidden_dim, self.data_dim))

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
        solution = diffrax.diffeqsolve(
            terms=ode_term,
            solver=diffrax.Heun(),
            t0=ts[0],
            t1=ts[-1],
            dt0=self.ode_solver_stepsize,
            y0=h0,
            saveat=saveat,
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        y = solution.ys[-1]
        y = self.dropout2(y, key=dropout_key2, inference=inference)
        (prediction,) = jnn.sigmoid(self.linear2(y))
        return prediction


class LogNeuralCDE(NeuralCDE):
    """Log-ODE CDE; only get_ode differs from NeuralCDE."""

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
            raise ValueError(
                "The Log-ODE method is only implemented for truncation depths one and two"
            )
        self.depth = depth
        self.hall_set = HallSet(data_dim, depth)

    def calc_logsigs(self, X):
        """Compute interval-wise log-signatures."""
        X = X.reshape(-1, self.stepsize, X.shape[-1])

        prepend = jnp.concatenate((jnp.zeros((1, X.shape[-1])), X[:-1, -1, :]))[
            :, None, :
        ]
        X = jnp.concatenate((prepend, X), axis=1)

        def logsig(x):
            logsig_val = flatten(log(signature(x, self.depth)))
            if self.depth == 1:
                return jnp.concatenate((jnp.array([0]), logsig_val))
            else:
                tensor_to_lie_map = self.hall_set.t2l_matrix(self.depth)
                return tensor_to_lie_map[:, 1:] @ logsig_val

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
        intervals = (
            jnp.arange(0, X.shape[0] + self.stepsize, self.stepsize) / X.shape[0]
        )

        def func(t, y, args):
            idx = jnp.searchsorted(intervals, t)
            logsig_t = logsigs[idx - 1]
            interval_length = intervals[idx] - intervals[idx - 1]
            if self.depth == 1:
                return self.depth_one_ode(y, logsig_t, interval_length)
            if self.depth == 2:
                return self.depth_two_ode(y, logsig_t, interval_length)

        return diffrax.ODETerm(func)



i = 8
last = 11
single = 3

features1 = np.load(
    f"/home/sichengyu/text/NCDE/SimplifiedProgram/autoencode/whisper1_12/2020/train_features_2020_single{single}.npy"
)
features2 = np.load(
    f"/home/sichengyu/text/NCDE/SimplifiedProgram/autoencode/whisper1_12/2020/test_features_2020_single{single}.npy"
)
labels1 = torch.load(
    f"/home/sichengyu/text/NCDE/feature_tensor/whisper1_12/2020/labels1_whisper_single{single}_0.3_30s_train.pt"
)
labels2 = torch.load(
    f"/home/sichengyu/text/NCDE/feature_tensor/whisper1_12/2020/labels2_whisper_single{single}_0.3_30s_test.pt"
)
indices_train = torch.load(
    f"/home/sichengyu/text/NCDE/feature_tensor/whisper1_12/2020/indices_whisper_single{single}_0.3_30s_train.pt"
)
indices_test = torch.load(
    f"/home/sichengyu/text/NCDE/feature_tensor/whisper1_12/2020/indices_whisper_single{single}_0.3_30s_test.pt"
)

features_np = features1
features_np_test = features2
labels1_np = labels1.detach().cpu().numpy()
labels2_np = labels2.detach().cpu().numpy()

features_jax = jnp.array(features_np)
features_jax_test = jnp.array(features_np_test)
labels_jax = jnp.array(labels1_np)
labels_jax_test = jnp.array(labels2_np)

num_zeros = jnp.sum(labels_jax_test == 1)
labels_jax_test


def preprocess_data(features):
    """Standardize over batch and time."""
    mean = features.mean((0, 1), keepdims=True)
    std = features.std((0, 1), keepdims=True)
    standardized_features = (features - mean) / (std + 1e-8)
    return standardized_features


def get_data(features):
    """Attach time channel and normalize features."""
    ts = jnp.linspace(0, 1, features.shape[1])
    ts1 = jnp.repeat(ts[None, :], features.shape[0], axis=0)

    normalized_features = preprocess_data(features)
    time_steps_expanded = ts1[:, :, None]
    features_with_time = jnp.concatenate(
        [time_steps_expanded, normalized_features], axis=2
    )
    return features_with_time


def count_audio_files(directory):
    """Count audio files recursively under a directory."""
    audio_extensions = (".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a", ".wma")
    audio_file_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_file_count += 1
    return audio_file_count


train_ccn = count_audio_files("")
train_cdn = count_audio_files("")
test_ccn = count_audio_files("")
test_cdn = count_audio_files("")


class Dataloader:
    """Simple infinite dataloader."""

    data: jnp.ndarray
    labels: jnp.ndarray
    size: int

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.size = len(data)

    def loop(self, batch_size, *, key=None):
        """Yield batches forever."""
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

    loss = -(
        y * jnp.log(pred_y_clipped)
        + (1 - y) * jnp.log(1 - pred_y_clipped)
    )
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
    preds_csv="predictions.csv",
):
    """
    Save scalar metrics and prediction lists to separate CSV files.
    Avoids duplicated rows for the same seed.
    """
    # ---- metrics ----
    if not os.path.exists(metrics_csv):
        df_metrics = pd.DataFrame(
            columns=[
                "seed",
                "acc_test",
                "f1",
                "precision",
                "recall",
                "acc_test_vote",
                "f1_vote",
                "precision_vote",
                "recall_vote",
            ]
        )
    else:
        df_metrics = pd.read_csv(metrics_csv)

    if seed in df_metrics["seed"].values:
        print(f"[INFO] Seed={seed} metrics already exist in {metrics_csv}, skip.")
    else:
        new_row_df = pd.DataFrame(
            [
                {
                    "seed": seed,
                    "acc_test": acc_test,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "acc_test_vote": acc_test_vote,
                    "f1_vote": f1_vote,
                    "precision_vote": precision_vote,
                    "recall_vote": recall_vote,
                }
            ]
        )
        df_metrics = pd.concat([df_metrics, new_row_df], ignore_index=True)
        df_metrics.to_csv(metrics_csv, index=False)
        print(f"[SUCCESS] Saved metrics for seed={seed} to {metrics_csv}.")

    # ---- predictions ----
    if not os.path.exists(preds_csv):
        df_preds = pd.DataFrame(
            columns=["seed", "test_predictions1", "test_predictions2"]
        )
    else:
        df_preds = pd.read_csv(preds_csv)

    if seed in df_preds["seed"].values:
        print(f"[INFO] Seed={seed} predictions already exist in {preds_csv}, skip.")
    else:
        test_preds1_str = str(test_predictions1)
        test_preds2_str = str(test_predictions2)

        new_row_preds_df = pd.DataFrame(
            [
                {
                    "seed": seed,
                    "test_predictions1": test_preds1_str,
                    "test_predictions2": test_preds2_str,
                }
            ]
        )
        df_preds = pd.concat([df_preds, new_row_preds_df], ignore_index=True)
        df_preds.to_csv(preds_csv, index=False)
        print(f"[SUCCESS] Saved predictions for seed={seed} to {preds_csv}.")


def get_trainable_params(model):
    """Filter trainable (inexact) arrays for optimisation."""
    return eqx.filter(model, eqx.is_inexact_array)


def train_model(
    model,
    num_steps=120,
    print_steps=20,
    batch_size=32,
    base_lr=3.5e-4,
    warmup_steps=24,
    weight_decay=0.0,
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
        alpha=0.01,
    )

    lr_schedule = optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps],
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

            if step == 0 or (step + 1) % print_steps == 0 or step == (
                steps_per_epoch - 1
            ):
                inference_model = eqx.nn.inference_mode(model)
                inference_model = eqx.Partial(inference_model, inference=True)

                for batch, data in zip(
                    range(1), train_dataloader.loop(train_dataloader.size)
                ):
                    X, y = data
                    keys = jax.random.split(jr.PRNGKey(0), X.shape[0])
                    output = jax.vmap(inference_model)(X, keys)
                    pre_train = output
                    train_acc = jnp.mean((output > 0.5) == (y == 1))

                for batch, data in zip(
                    range(1), test_dataloader.loop(test_dataloader.size)
                ):
                    X, y = data
                    keys = jax.random.split(jr.PRNGKey(0), X.shape[0])
                    output = jax.vmap(inference_model)(X, keys)
                    test_acc = jnp.mean((output > 0.5) == (y == 1))
                if step == steps_per_epoch - 1:
                    pre_test = output

                elapsed_time = time.time() - start_time
                print(
                    f"Step: {step + 1}, Loss: {loss}, "
                    f"Train Acc: {train_acc}, Test Acc: {test_acc}, "
                    f"Time: {elapsed_time:.4f} s"
                )

                steps.append(step + 1)

        # aggregate train predictions per original audio
        audio_segments_train = {}
        for idx, pred in zip(indices_train[:, 0], pre_train):
            if idx.size == 1:
                idx = int(idx.item())
            else:
                raise ValueError(f"Unexpected idx size: {idx.size}, idx: {idx}")
            if idx in audio_segments_train:
                audio_segments_train[idx].append(pred)
            else:
                audio_segments_train[idx] = [pred]

        audio_predictions_train = {
            idx: jnp.mean(jnp.array(preds))
            for idx, preds in audio_segments_train.items()
        }
        predictions1 = jnp.array(list(audio_predictions_train.values()))
        train_predictions1 = [
            (idx, 1 if pred >= 0.5 else 0)
            for idx, pred in audio_predictions_train.items()
        ]

        # aggregate test predictions per original audio
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

        audio_predictions_test = {
            idx: jnp.mean(jnp.array(preds))
            for idx, preds in audio_segments_test.items()
        }
        audio_predictions_test_vote = {
            idx: 1
            if jnp.sum(jnp.array(preds) > 0.5)
            > jnp.sum(jnp.array(preds) <= 0.5)
            else 0
            for idx, preds in audio_segments_test.items()
        }

        correct_predictions_test = 0
        predict_label = []
        predict_label_vote = []

        for idx, pred in audio_predictions_test.items():
            label = labels_test[idx]
            predict_label.append(pred > 0.5)
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

        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("acc_test:", acc_test)
        print("acc_test_vote:", acc_test_vote)

        test_accs.append(acc_test)
        test_accs_vote.append(acc_test_vote)

        if epoch == total_epochs - 1:
            predictions_list_test = []
            for idx, preds in audio_segments_test.items():
                mean_pred = jnp.mean(jnp.array(preds))
                predictions_list_test.append((idx, mean_pred))

            predictions_list_test.sort(key=lambda x: x[0])

            for idx, mean_pred in predictions_list_test:
                print(f"Audio segment test {idx}: prediction value {mean_pred}")

            test_predictions1 = [
                (idx, 1 if pred >= 0.5 else 0) for idx, pred in predictions_list_test
            ]
            test_predictions2 = [
                (idx, 1 if pred == 1 else 0)
                for idx, pred in audio_predictions_test_vote.items()
            ]
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
                metrics_csv=(
                    f"/home/sichengyu/text/NCDE/SimplifiedProgram/solution/"
                    f"whisper1-12/2020/whisper_single{single}_log_50seed_"
                    f"acc_h128_v512_0norm_ode250_step50.csv"
                ),
                preds_csv=(
                    f"/home/sichengyu/text/NCDE/SimplifiedProgram/solution/"
                    f"whisper1-12/2020/whisepr_single{single}_log_50seed_"
                    f"predict_h128_v512_0norm_ode250_step50.csv"
                ),
            )

    return (
        acc_test,
        acc_test_vote,
        test_accs,
        test_accs_vote,
        train_predictions1,
        test_predictions1,
        test_predictions2,
    )


warnings.filterwarnings("ignore", category=FutureWarning)


def train_with_seeds(seeds):
    """Train with multiple seeds and aggregate results."""
    global hyperparameters, all_train_predictions, all_test_predictions_vote
    global all_test_predictions, acc_seeds

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

            (
                acc_test,
                acc_test_vote,
                test_accs,
                test_accs_vote,
                train_predictions1,
                test_predictions1,
                test_predictions2,
            ) = train_model(LogNCDE_Depth2, key=trainkey, seed=seed)

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
            placeholder_accs = (
                np.zeros(len(test_accs)) if "test_accs" in locals() else np.zeros(10)
            )
            all_test_accuracies.append(placeholder_accs)
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
    df_test = pd.DataFrame(
        all_test_accuracies, columns=epochs, index=[f"Seed_{s}" for s in seeds]
    )
    df_test_vote = pd.DataFrame(
        all_test_accuracies_vote, columns=epochs, index=[f"Seed_{s}" for s in seeds]
    )
    print("\nTest accuracies per epoch per seed:")
    print(df_test)
    print("\nTest accuracies (vote) per epoch per seed:")
    print(df_test_vote)
    return df_test


def vote_and_evaluate():
    """Aggregate predictions across seeds and compute metrics."""
    train_votes = aggregate_votes(all_train_predictions)
    accuracy = evaluate_accuracy(train_votes, labels_train, "Train")

    test_votes = aggregate_votes(all_test_predictions)
    accuracy_test = evaluate_accuracy(test_votes, labels_test, "Test")
    test_votes = jnp.array(list(test_votes.values()))

    test_vote_votes = aggregate_votes(all_test_predictions_vote)
    accuracy_vote_test = evaluate_accuracy(
        test_vote_votes, labels_test, "Test_vote"
    )
    test_vote_votes = jnp.array(list(test_vote_votes.values()))

    precision = precision_score(labels_test, test_votes)
    print(f"Precision: {precision}")
    recall = recall_score(labels_test, test_votes)
    print(f"Recall: {recall}")
    f1 = f1_score(labels_test, test_votes)
    print(f"F1 Score: {f1}")

    precision = precision_score(labels_test, test_vote_votes)
    print(f"Precision_vote: {precision}")
    recall = recall_score(labels_test, test_vote_votes)
    print(f"Recall_vote: {recall}")
    f1 = f1_score(labels_test, test_vote_votes)
    print(f"F1 Score_vote: {f1}")
    print("test_vote_votes:", test_vote_votes)


def aggregate_votes(predictions_list_all_seeds):
    """Majority vote across seeds for each audio index."""
    aggregated_predictions = {}
    for idx in range(len(predictions_list_all_seeds[0])):
        preds = [
            predictions_list_all_seeds[seed_idx][idx][1]
            for seed_idx in range(len(predictions_list_all_seeds))
        ]
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
    print(f"{dataset_name} accuracy (after voting): {accuracy:.2f}")
    return accuracy


hidden_dim = 128
data_dim = 33
label_dim = 1
vf_hidden_dim = 512
vf_num_hidden = 3
ode_solver_stepsize = 1 / 250
stepsize = 60
num_seeds = 9
seeds = np.arange(1001, 1006).tolist()
print("Seeds:", seeds)

train_with_seeds(seeds)


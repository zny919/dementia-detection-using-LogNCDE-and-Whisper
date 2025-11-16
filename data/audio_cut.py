# Environment and random seeds
import os, random, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('CUDA_VISIBLE_DEVICES', '5')
os.environ["PYTHONHASHSEED"] = "400"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

random.seed(400)
np.random.seed(400)

import torch
torch.manual_seed(400)
torch.cuda.manual_seed_all(400)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Other imports
import re
from pathlib import Path
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel

# Basic config
MODEL_NAME = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_KEY = "Pitt"   # "NCMMSC2021" / "ADReSSo21" / "ADReSS-IS2020" / "Pitt"

sampling_rate = 16000
SEGMENT_SEC   = 30
segment_length1 = SEGMENT_SEC * sampling_rate
overlap_rate1   = 0.3

# Single-layer extraction (takes priority if list is non-empty)
SINGLE_LAYER_LIST = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
NORM_FOR_SINGLE   = "ln_post"    # None / "layernorm" / "std" / "ln_post"

# Last-k fusion (used only if SINGLE_LAYER_LIST is empty)
LAST_K_LIST       = []
DROP_INPUT_LAYER  = True
LAYER_F1          = None         # if provided, use F1-based weights
F1_TEMP           = 12
NORM_BEFORE_FUSE  = "ln_post"    # None / "layernorm" / "std" / "ln_post"

TOP_LAYER_INDEX = 12

# Output directory
save_path = "/home/sichengyu/text/NCDE/feature_tensor/whisper1_12/Pitt"

# Dataset paths
DATASETS = {
    "NCMMSC2021": {
        "train_cc": "/home/sichengyu/Downloads/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/train/HC",
        "train_cd": "/home/sichengyu/Downloads/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/train/AD_combine",
        "test_cc" : "/home/sichengyu/Downloads/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/test_have_label/HC",
        "test_cd" : "/home/sichengyu/Downloads/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/test_have_label/AD_combine",
    },
    "ADReSSo21": {
        "train_cc": "/home/sichengyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/cn",
        "train_cd": "/home/sichengyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/ad",
        "test_cc" : "/home/sichengyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/Control",
        "test_cd" : "/home/sichengyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/ProbableAD",
    },
    "ADReSS-IS2020": {
        "train_cc": "/home/sichengyu/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc",
        "train_cd": "/home/sichengyu/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd",
        "test_cc" : "/home/sichengyu/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cc_audio",
        "test_cd" : "/home/sichengyu/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cd_audio",
    },
    "Pitt": {
        "train_cc": "/home/sichengyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cc_enhence",
        "train_cd": "/home/sichengyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cd_enhence",
        "test_cc" : "/home/sichengyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cc_enhence",
        "test_cd" : "/home/sichengyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cd_enhence",
    },
}
assert DATASET_KEY in DATASETS, f"Unknown DATASET_KEY={DATASET_KEY}"

# Helpers
def natural_key(p: str):
    s = Path(p).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def weights_from_f1(f1_list, temp=8.0):
    import numpy as _np
    f1 = _np.asarray(f1_list, dtype=_np.float64)
    fmin, fmax = float(f1.min()), float(f1.max())
    if abs(fmax - fmin) < 1e-12:
        return _np.ones_like(f1) / len(f1)
    s = (f1 - fmin) / (fmax - fmin)
    e = _np.exp(temp * s)
    return e / (e.sum() + 1e-12)

def apply_norm(h: torch.Tensor, mode: str, ln_post=None, skip_ln_post: bool=False):
    """Normalize [1, T, C] according to mode."""
    if mode is None:
        return h
    if mode == "layernorm":
        return F.layer_norm(h, h.shape[-1:])
    if mode == "std":
        std = h.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return h / std
    if mode in ("ln_post", "whisper_ln"):
        if skip_ln_post:
            return h
        if ln_post is None:
            raise ValueError("ln_post is None, but 'ln_post/whisper_ln' was requested")
        return ln_post(h)
    raise ValueError(f"Unknown norm mode: {mode}")

def stack_with_norm(hs_list, mode, ln_post=None, skip_last_ln_post=False):
    if mode is None:
        return torch.stack(hs_list, dim=0)
    out, L = [], len(hs_list)
    for i, h in enumerate(hs_list):
        skip = (skip_last_ln_post and i == L - 1 and mode in ("ln_post", "whisper_ln"))
        out.append(apply_norm(h, mode, ln_post=ln_post, skip_ln_post=skip))
    return torch.stack(out, dim=0)

def _model_tag_from_name(model_name: str):
    tail = model_name.split("/")[-1]
    return tail.replace("whisper-", "whisper_")

# Whisper model and final layer norm
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model     = WhisperModel.from_pretrained(MODEL_NAME)
model.eval().to(DEVICE)
LN_POST = model.encoder.layer_norm

with torch.no_grad():
    w = LN_POST.weight.detach().cpu().float()
    b = LN_POST.bias.detach().cpu().float()
    print(f"[LN_POST] eps={LN_POST.eps}  elementwise_affine={LN_POST.elementwise_affine}")
    print(f"[LN_POST] weight mean={w.mean():.6f} std={w.std():.6f} min={w.min():.6f} max={w.max():.6f}")
    print(f"[LN_POST] bias   mean={b.mean():.6f} std={b.std():.6f} min={b.min():.6f} max={b.max():.6f}")

# Per-file segment extraction
@torch.inference_mode()
def get_segment_features(audio_file: str,
                         audio_file_index: int,
                         single_layer=None,
                         norm_single=None,
                         last_k=None,
                         drop_input_layer=True,
                         layer_f1=None,
                         f1_temp=12,
                         norm_mode=None,
                         segment_length=segment_length1,
                         sampling_rate=16000,
                         overlap_rate=overlap_rate1):
    import numpy as _np
    y, _ = librosa.load(audio_file, sr=sampling_rate, mono=True, dtype=_np.float32)

    step_size = int(segment_length * (1 - overlap_rate))
    total_segments = int(len(y) / step_size) + (1 if len(y) % step_size > overlap_rate * segment_length else 0)

    seg_features, seg_indices = [], []
    warned_double_ln_top = False

    for i in range(total_segments):
        start, end = i * step_size, i * step_size + segment_length
        segment = y[start:end]

        # Pad last short segment (except the first one) using previous samples
        if len(segment) < segment_length and i > 0:
            padding_needed = segment_length - len(segment)
            prev_end = start + segment_length - step_size
            padding_start = max(0, prev_end - padding_needed)
            padding_values = y[padding_start:prev_end]
            segment = _np.concatenate((segment, padding_values))

        inp = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inp.input_features.to(DEVICE)

        enc_out = model.encoder(input_features=input_features,
                                output_hidden_states=True, return_dict=True)
        hs_all = list(enc_out.hidden_states)  # len = 1 + L
        hs = hs_all[1:] if (drop_input_layer and len(hs_all) >= 2) else hs_all

        # Check if the top layer already matches last_hidden_state (post-LN)
        try:
            already_post = torch.allclose(hs[-1], enc_out.last_hidden_state, atol=1e-5, rtol=1e-5)
        except Exception:
            already_post = True

        if single_layer is not None:
            assert 1 <= int(single_layer) <= len(hs), f"SINGLE_LAYER={single_layer} out of range"
            is_top = (int(single_layer) == len(hs))
            if (not warned_double_ln_top and is_top and already_post and norm_single in ("ln_post", "whisper_ln")):
                print(f"[WARN] {Path(audio_file).name}: top layer already post-LN, skipping extra ln_post.")
                warned_double_ln_top = True

            h = hs[int(single_layer) - 1]
            is_top = (int(single_layer) == len(hs))
            skip_ln = (norm_single in ("ln_post", "whisper_ln")) and (is_top or int(single_layer) == TOP_LAYER_INDEX)
            fused = apply_norm(h, norm_single, ln_post=LN_POST, skip_ln_post=skip_ln)

        else:
            if layer_f1 is not None:
                assert len(layer_f1) == len(hs), "layer_f1 length must match number of layers"
                if (not warned_double_ln_top and norm_mode in ("ln_post", "whisper_ln") and already_post):
                    print(f"[WARN] {Path(audio_file).name}: fusion uses ln_post but top layer is already post-LN; skipping top ln_post.")
                    warned_double_ln_top = True
                H = stack_with_norm(hs, norm_mode, ln_post=LN_POST, skip_last_ln_post=already_post)
                w_np = weights_from_f1(layer_f1, temp=f1_temp)
                w = torch.tensor(w_np, dtype=H.dtype, device=H.device).view(-1, 1, 1, 1)
                fused = (w * H).sum(dim=0)
            else:
                assert last_k is not None and 1 <= last_k <= len(hs), "invalid last_k"
                selected = hs[-last_k:]
                if (not warned_double_ln_top and norm_mode in ("ln_post", "whisper_ln") and already_post):
                    print(f"[WARN] {Path(audio_file).name}: last-k fusion uses ln_post but top layer is already post-LN; skipping top ln_post.")
                    warned_double_ln_top = True
                # If hs has TOP_LAYER_INDEX layers, also skip ln_post on the last layer
                skip_last = already_post or (len(hs) == TOP_LAYER_INDEX)
                H = stack_with_norm(selected, norm_mode, ln_post=LN_POST, skip_last_ln_post=skip_last)
                fused = H[0] if last_k == 1 else H.mean(dim=0)

        fused = fused.squeeze(0).detach().cpu()  # [T, C]
        seg_features.append(fused)
        seg_indices.append((audio_file_index, i))

    return seg_features, seg_indices

# Directory-level extraction
def extract_dir(audio_path: str, **kwargs):
    files = sorted(
        [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith(".wav")],
        key=natural_key
    )
    features, indices = [], []
    for idx, fp in enumerate(tqdm(files, desc=f"[DIR] {audio_path}", ncols=90)):
        seg_feats, seg_idx = get_segment_features(fp, idx, **kwargs)
        features.extend(seg_feats)
        indices.extend(seg_idx)
    return features, indices

OV_STR  = "0.3"
SEC_STR = f"{SEGMENT_SEC}s"

# Run extraction and save tensors for one config
def run_and_save_one_config(*, dataset_paths, single_layer=None, last_k=None,
                            norm_single=None, norm_mode=None, drop_input_layer=True,
                            layer_f1=None, f1_temp=12):
    feat_cc_tr, idx_cc_tr = extract_dir(
        dataset_paths["train_cc"],
        single_layer=single_layer, norm_single=norm_single,
        last_k=last_k, drop_input_layer=drop_input_layer,
        layer_f1=layer_f1, f1_temp=f1_temp, norm_mode=norm_mode
    )
    feat_cd_tr, idx_cd_tr = extract_dir(
        dataset_paths["train_cd"],
        single_layer=single_layer, norm_single=norm_single,
        last_k=last_k, drop_input_layer=drop_input_layer,
        layer_f1=layer_f1, f1_temp=f1_temp, norm_mode=norm_mode
    )
    feat_cc_te, idx_cc_te = extract_dir(
        dataset_paths["test_cc"],
        single_layer=single_layer, norm_single=norm_single,
        last_k=last_k, drop_input_layer=drop_input_layer,
        layer_f1=layer_f1, f1_temp=f1_temp, norm_mode=norm_mode
    )
    feat_cd_te, idx_cd_te = extract_dir(
        dataset_paths["test_cd"],
        single_layer=single_layer, norm_single=norm_single,
        last_k=last_k, drop_input_layer=drop_input_layer,
        layer_f1=layer_f1, f1_temp=f1_temp, norm_mode=norm_mode
    )

    features_cc      = torch.stack(feat_cc_tr, dim=0)
    features_cd      = torch.stack(feat_cd_tr, dim=0)
    features_cc_test = torch.stack(feat_cc_te, dim=0)
    features_cd_test = torch.stack(feat_cd_te, dim=0)

    features1 = torch.cat([features_cc, features_cd], dim=0)             # train
    features2 = torch.cat([features_cc_test, features_cd_test], dim=0)   # test

    labels_cc      = torch.zeros(len(feat_cc_tr))
    labels_cd      = torch.ones(len(feat_cd_tr))
    labels_cc_test = torch.zeros(len(feat_cc_te))
    labels_cd_test = torch.ones(len(feat_cd_te))
    labels1 = torch.cat([labels_cc, labels_cd], dim=0)
    labels2 = torch.cat([labels_cc_test, labels_cd_test], dim=0)

    idx_cc_tr = np.array(idx_cc_tr)
    idx_cd_tr = np.array(idx_cd_tr)
    if len(idx_cc_tr) > 0 and len(idx_cd_tr) > 0:
        max_idx = idx_cc_tr[:, 0].max()
        idx_cd_tr[:, 0] += (max_idx + 1)
    indices_train = np.vstack([idx_cc_tr, idx_cd_tr]) if len(idx_cd_tr) > 0 else idx_cc_tr

    idx_cc_te = np.array(idx_cc_te)
    idx_cd_te = np.array(idx_cd_te)
    if len(idx_cc_te) > 0 and len(idx_cd_te) > 0:
        max_idx = idx_cc_te[:, 0].max()
        idx_cd_te[:, 0] += (max_idx + 1)
    indices_test = np.vstack([idx_cc_te, idx_cd_te]) if len(idx_cd_te) > 0 else idx_cc_te

    if single_layer is not None:
        file_name  = f"feature_whisper_single{single_layer}_{OV_STR}_{SEC_STR}_train.pt"
        file_name2 = f"feature_whisper_single{single_layer}_{OV_STR}_{SEC_STR}_test.pt"
        file_name3 = f"labels1_whisper_single{single_layer}_{OV_STR}_{SEC_STR}_train.pt"
        file_name4 = f"labels2_whisper_single{single_layer}_{OV_STR}_{SEC_STR}_test.pt"
        file_name5 = f"indices_whisper_single{single_layer}_{OV_STR}_{SEC_STR}_train.pt"
        file_name6 = f"indices_whisper_single{single_layer}_{OV_STR}_{SEC_STR}_test.pt"
    else:
        file_name  = f"feature_whisper_last{last_k}_{OV_STR}_{SEC_STR}_train.pt"
        file_name2 = f"feature_whisper_last{last_k}_{OV_STR}_{SEC_STR}_test.pt"
        file_name3 = f"labels1_whisper_last{last_k}_{OV_STR}_{SEC_STR}_train.pt"
        file_name4 = f"labels2_whisper_last{last_k}_{OV_STR}_{SEC_STR}_test.pt"
        file_name5 = f"indices_whisper_last{last_k}_{OV_STR}_{SEC_STR}_train.pt"
        file_name6 = f"indices_whisper_last{last_k}_{OV_STR}_{SEC_STR}_test.pt"

    full_path  = os.path.join(save_path, file_name)
    full_path2 = os.path.join(save_path, file_name2)
    full_path3 = os.path.join(save_path, file_name3)
    full_path4 = os.path.join(save_path, file_name4)
    full_path5 = os.path.join(save_path, file_name5)
    full_path6 = os.path.join(save_path, file_name6)

    torch.save(features1, full_path)
    torch.save(features2, full_path2)
    torch.save(labels1,   full_path3)
    torch.save(labels2,   full_path4)
    torch.save(indices_train, full_path5)
    torch.save(indices_test,  full_path6)

    print(f"[SAVE] -> {save_path}")
    print("  -", full_path)
    print("  -", full_path2)
    print("  -", full_path3)
    print("  -", full_path4)
    print("  -", full_path5)
    print("  -", full_path6)

# Main entry
paths = DATASETS[DATASET_KEY]
print(f"[DATASET] {DATASET_KEY}")
for k, v in paths.items():
    print(f"  {k}: {v}")

if len(SINGLE_LAYER_LIST) > 0:
    print(f"\n[RUN] Single-layer batch: {SINGLE_LAYER_LIST}, NORM_FOR_SINGLE={NORM_FOR_SINGLE}")
    for lyr in SINGLE_LAYER_LIST:
        print(f"\n>>> [Single Layer] layer={lyr}")
        run_and_save_one_config(
            dataset_paths=paths,
            single_layer=int(lyr),
            last_k=None,
            norm_single=NORM_FOR_SINGLE,
            norm_mode=None,
            drop_input_layer=DROP_INPUT_LAYER,
            layer_f1=None,
            f1_temp=F1_TEMP,
        )
elif len(LAST_K_LIST) > 0:
    print(f"\n[RUN] Last-k fusion batch: {LAST_K_LIST}, NORM_BEFORE_FUSE={NORM_BEFORE_FUSE}")
    for k in LAST_K_LIST:
        print(f"\n>>> [Last-K Fuse] K={k}")
        run_and_save_one_config(
            dataset_paths=paths,
            single_layer=None,
            last_k=int(k),
            norm_single=None,
            norm_mode=NORM_BEFORE_FUSE,
            drop_input_layer=DROP_INPUT_LAYER,
            layer_f1=LAYER_F1,
            f1_temp=F1_TEMP,
        )
else:
    raise ValueError("Provide at least one list in SINGLE_LAYER_LIST or LAST_K_LIST.")

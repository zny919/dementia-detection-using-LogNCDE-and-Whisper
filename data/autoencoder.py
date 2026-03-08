import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import torch
import torch.nn as nn
import torch.optim as optim

last = 1
single = 12 
base_path = "/path/to/features/whisper_medium_12/NCMMSC2021"

features1 = torch.load(f"{base_path}/feature_whisper_last{last}_0.3_30s_train.pt")
features2 = torch.load(f"{base_path}/feature_whisper_last{last}_0.3_30s_test.pt")
labels1 = torch.load(f"{base_path}/labels1_whisper_last{last}_0.3_30s_train.pt")
labels2 = torch.load(f"{base_path}/labels2_whisper_last{last}_0.3_30s_test.pt")
indices_train = torch.load(f"{base_path}/indices_whisper_last{last}_0.3_30s_train.pt")
indices_test = torch.load(f"{base_path}/indices_whisper_last{last}_0.3_30s_test.pt")

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(400)

def preprocess_data(features):
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    
    """
    ADReSSo and NCMMSC datasets use this code:
    features1 = preprocess_data(features1)
    features2 = preprocess_data(features2)
    """
    standardized_features = (features - mean) / (std + 1e-8)
    return standardized_features

features1 = preprocess_data(features1)
features2 = preprocess_data(features2)

import pandas as pd

train_features = features1.numpy()
test_features = features2.numpy()

train_mean = np.mean(train_features)
train_std = np.std(train_features)
test_mean = np.mean(test_features)
test_std = np.std(test_features)

print("Overall mean and std:")
print(f"Train mean: {train_mean:.6f}, std: {train_std:.6f}")
print(f"Test mean: {test_mean:.6f}, std: {test_std:.6f}")

import matplotlib.pyplot as plt


class ConvAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, latent_dim, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, input_dim, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

def _to_device_whole_tensor(x, device):
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not torch.is_floating_point(x):
        x = x.float()
    return x.to(device, non_blocking=True)

def train_autoencoder_with_early_stopping(
    model, train_data, val_data, epochs=50, lr=0.001, batch_size=64, patience=5, min_delta=0.001
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_tensor = _to_device_whole_tensor(train_data, device)
    val_tensor   = _to_device_whole_tensor(val_data, device)

    train_ds = torch.utils.data.TensorDataset(train_tensor)
    val_ds   = torch.utils.data.TensorDataset(val_tensor)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    model.train()
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        for batch in train_loader:
            batch = batch[0]
            optimizer.zero_grad(set_to_none=True)
            _, reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_loss = epoch_train_loss / max(len(train_loader), 1)
        train_losses.append(train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch[0]
                _, reconstructed_val = model(val_batch)
                val_loss = criterion(reconstructed_val, val_batch)
                epoch_val_loss += val_loss.item()

        val_loss = epoch_val_loss / max(len(val_loader), 1)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        model.train()

    print(f"Early stopping triggered at epoch {epoch + 1}. Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f}")
    return model, train_losses, val_losses, best_epoch

def plot_losses(train_losses, val_losses=None):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    if val_losses is not None:
        plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("Training and Validation Reconstruction Loss")
    plt.legend()
    plt.show()

def extract_features(model, data, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    if isinstance(data, (list, tuple)):
        data = data[0]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if not torch.is_floating_point(data):
        data = data.float()
    data = data.to(device, non_blocking=True)

    with torch.no_grad():
        latent, _ = model(data)

    latent_np = latent.detach().cpu().numpy()

    if save_path:
        np.save(save_path, latent_np)
        print(f"Low-dimensional features saved to: {save_path}")

    return latent_np

if __name__ == "__main__":
    train_data = features1 
    test_data = features2 

    train_data = train_data.permute(0, 2, 1)
    test_data = test_data.permute(0, 2, 1)

    folder_path = "/path/to/save/whisper_medium_1_12_auto256/NCMMSC2021"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

    autoencoder = ConvAutoEncoder(input_dim=1024, latent_dim=64)

    autoencoder, train_losses, val_losses, best_epoch = train_autoencoder_with_early_stopping(
        autoencoder, train_data, test_data, epochs=80, lr=0.001, batch_size=16
    )

    plot_losses(train_losses, val_losses)
    
    train_features = extract_features(autoencoder, train_data)
    train_features = np.transpose(train_features, (0, 2, 1))
    train_file_path = os.path.join(folder_path, f"train_features_NCMMSC2021_last{last}.npy")
    np.save(train_file_path, train_features)
    print(f"Train features shape: {train_features.shape}")
    print(f"Train features saved at: {train_file_path}")

    test_features = extract_features(autoencoder, test_data)
    test_features = np.transpose(test_features, (0, 2, 1))
    test_file_path = os.path.join(folder_path, f"test_features_NCMMSC2021_last{last}.npy")
    np.save(test_file_path, test_features)

    print(f"Test features shape: {test_features.shape}")
    print(f"Test features saved at: {test_file_path}")

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

last = 1  #last k layer fusion
single = 1  # single layer of whisper

features1 = torch.load(
    f"/home/sichengyu/text/NCDE/feature_tensor/whisper_medium_12/NCMMSC2021/feature_whisper_last{last}_0.3_30s_train.pt"
)
features2 = torch.load(
    f"/home/sichengyu/text/NCDE/feature_tensor/whisper_medium_12/NCMMSC2021/feature_whisper_last{last}_0.3_30s_test.pt"
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(400)

def preprocess_data(features: torch.Tensor) -> torch.Tensor:
    """Standardize features along the last dimension."""
    mean = features.mean(axis=2, keepdims=True)
    std = features.std(axis=2, keepdims=True)
    standardized = (features - mean) / (std + 1e-8)
    return standardized

''' ADReSSo and NCMMSC datasets use this codes
 features1 = preprocess_data(features1)
 features2 = preprocess_data(features2)
'''

# Inspect global statistics
train_features_np = features1.numpy()
test_features_np = features2.numpy()

train_mean = np.mean(train_features_np)
train_std = np.std(train_features_np)
test_mean = np.mean(test_features_np)
test_std = np.std(test_features_np)

class ConvAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, latent_dim, kernel_size=3, padding=1),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, input_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


def _to_device_whole_tensor(x, device: torch.device) -> torch.Tensor:
    """Move the whole tensor to device once to avoid per-batch H2D copies."""
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not torch.is_floating_point(x):
        x = x.float()
    return x.to(device, non_blocking=True)


def train_autoencoder(
    model: nn.Module,
    train_data,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 64,
):
    """Train autoencoder on train_data only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_tensor = _to_device_whole_tensor(train_data, device)

    train_ds = torch.utils.data.TensorDataset(train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            _, reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(len(train_loader), 1)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.6f}")

    return model, train_losses


def plot_losses(train_losses):
    """Plot training reconstruction loss."""
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("Training Reconstruction Loss")
    plt.legend()
    plt.show()


def extract_features(model: nn.Module, data, save_path: str | None = None):
    """Extract latent features using a trained autoencoder."""
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

    if save_path is not None:
        np.save(save_path, latent_np)
        print(f"Low-dimensional features saved to: {save_path}")

    return latent_np


if __name__ == "__main__":
    train_data = features1
    test_data = features2

    # [N, T, C] -> [N, C, T] for Conv1d
    train_data = train_data.permute(0, 2, 1)
    test_data = test_data.permute(0, 2, 1)

    folder_path = "whisper_medium_1_12_auto256/NCMMSC2021"
    os.makedirs(folder_path, exist_ok=True)
    print(f"Using output folder: {folder_path}")

    input_dim = train_data.shape[1]
    autoencoder = ConvAutoEncoder(input_dim=input_dim, latent_dim=64)

    autoencoder, train_losses = train_autoencoder(
        autoencoder, train_data, epochs=80, lr=0.001, batch_size=16
    )

    plot_losses(train_losses)

    # Train features
    train_features = extract_features(autoencoder, train_data)
    train_features = np.transpose(train_features, (0, 2, 1))
    train_file_path = os.path.join(
        folder_path, f"train_features_NCMMSC2021_last{last}.npy"
    )
    np.save(train_file_path, train_features)
    print(f"Train features shape: {train_features.shape}")
    print(f"Train features saved at: {train_file_path}")

    # Test features
    test_features = extract_features(autoencoder, test_data)
    test_features = np.transpose(test_features, (0, 2, 1))
    test_file_path = os.path.join(
        folder_path, f"test_features_NCMMSC2021_last{last}.npy"
    )
    np.save(test_file_path, test_features)
    print(f"Test features saved at: {test_file_path}")


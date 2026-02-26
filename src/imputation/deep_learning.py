from typing import Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseImputer
from tqdm import trange



class _MLPAutoencoder(nn.Module):
    """
    Simple MLP Autoencoder:
    Encoder: input_dim -> hidden_dims...
    Decoder: ... -> input_dim
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.0):
        super().__init__()

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU())
            if dropout and dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        rev = list(hidden_dims)[::-1]
        prev = rev[0] if len(rev) > 0 else input_dim
        for h in rev[1:]:
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU())
            if dropout and dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            prev = h

        # Output layer back to input_dim (no ReLU at end; scaled values can be negative)
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class DenoisingAutoencoder(BaseImputer):
    """
    Denoising Autoencoder (DAE) Imputer for tabular *numeric* data.

    Key ideas:
    - Standardize features (StandardScaler) for stable training.
    - Build an "observed mask" for where values are NOT missing.
    - Fill NaNs with per-feature means as a *starting point*.
    - During training, additionally corrupt (mask) random observed inputs ("denoising").

    Assumes X is numeric. Categorical features must be encoded beforehand.
    """

    def __init__(
        self,
        name: str = "DAE",
        hidden_dims: Sequence[int] = (128, 64),
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        corruption_rate: float = 0.2,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        device: Optional[str] = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        super().__init__(name)
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.corruption_rate = float(corruption_rate)
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[_MLPAutoencoder] = None
        self.feature_means_scaled_: Optional[np.ndarray] = None  # means in scaled space

    @staticmethod
    def _as_numpy(X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, Optional[pd.Index], Optional[pd.Index]]:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32), X.columns, X.index
        return np.asarray(X, dtype=np.float32), None, None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None) -> "DenoisingAutoencoder":
        # Reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        X_np, cols, idx = self._as_numpy(X)

        # Mask: True where observed (not NaN)
        obs_mask = ~np.isnan(X_np)

        # Fill NaNs with per-feature mean (original space) so scaler can fit safely
        col_means = np.nanmean(X_np, axis=0)
        X_filled = np.where(obs_mask, X_np, col_means).astype(np.float32)

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_filled).astype(np.float32)

        # Store per-feature mean in scaled space (handy default fill later)
        self.feature_means_scaled_ = X_scaled.mean(axis=0).astype(np.float32)

        # Torch tensors
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        m_tensor = torch.tensor(obs_mask.astype(np.float32), dtype=torch.float32)

        dataset = TensorDataset(x_tensor, m_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        # Model
        input_dim = X_scaled.shape[1]
        self.model = _MLPAutoencoder(input_dim=input_dim, hidden_dims=self.hidden_dims, dropout=self.dropout).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.model.train()

        epoch_bar = trange(1, self.epochs + 1, desc=f"[{self.name}] training", disable=not self.verbose)

        for epoch in epoch_bar:
            total_loss = 0.0
            n_batches = 0

            for xb, mb in loader:
                xb = xb.to(self.device)  # scaled inputs
                mb = mb.to(self.device)  # observed mask (1 observed, 0 missing)

                # ----- DENOISING STEP -----
                # We only want to corrupt observed cells (not already-missing cells).
                # corruption_mask: 1 where we corrupt, 0 otherwise
                rand = torch.rand_like(xb)
                corruption_mask = ((rand < self.corruption_rate) * (mb > 0.5)).float()

                # Create noisy input by "dropping" corrupted values to 0
                xb_noisy = xb * (1.0 - corruption_mask)

                # Forward
                pred = self.model(xb_noisy)

                # ----- MASKED LOSS -----
                # Compute error only where values are truly observed (mb == 1).
                diff = (pred - xb) * mb
                loss = torch.sum(diff * diff) / torch.sum(mb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if self.verbose and (epoch == 1 or epoch % 20 == 0 or epoch == self.epochs):
                print(f"[{self.name}] epoch {epoch:>4}/{self.epochs}  loss={total_loss / max(n_batches,1):.6f}")

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        if self.model is None or self.scaler is None or self.feature_means_scaled_ is None:
            raise RuntimeError("DAE is not fitted yet. Call fit() first.")

        X_np, cols, idx = self._as_numpy(X)
        obs_mask = ~np.isnan(X_np)

        # Fill NaNs so we can scale. (This is just a temporary placeholder.)
        col_means = np.nanmean(X_np, axis=0)
        X_filled = np.where(obs_mask, X_np, col_means).astype(np.float32)

        # Scale using trained scaler
        X_scaled = self.scaler.transform(X_filled).astype(np.float32)

        # Replace missing cells in scaled space with learned feature means (better default than arbitrary 0)
        X_scaled = np.where(obs_mask, X_scaled, self.feature_means_scaled_).astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            pred_scaled = self.model(xb).cpu().numpy().astype(np.float32)

        # Only impute the originally missing cells
        X_imputed_scaled = np.where(obs_mask, X_scaled, pred_scaled)

        # Back to original scale
        X_imputed = self.scaler.inverse_transform(X_imputed_scaled).astype(np.float32)

        if cols is not None:
            return pd.DataFrame(X_imputed, columns=cols, index=idx)
        return X_imputed

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y=None) -> Union[np.ndarray, pd.DataFrame]:
        return self.fit(X, y=y).transform(X)

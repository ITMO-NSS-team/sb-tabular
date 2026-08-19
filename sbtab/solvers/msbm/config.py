import torch

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

FB = Literal["f", "b"]

@dataclass
class MixedSBMConfig:
    """
    Configuration for MSBM solver.

    Attributes:
        fb_sequence: Sequence of phases "forward" and "backward" encoded as "f" and "b". Always starts and ends with "b".
        cat_emb_dim: Dimensionality of categorical embeddings.
        hidden_dim: Hidden layer's dimensionality.
        time_dim: Time dimensionality.
        n_layers: Number of layers in MSBM.
        dropout: Dropout rate.
        weight_decay: Weight decay of the backbone model.
        num_steps: Number of steps in one outer loop of MSBM.
        eps: Updater clamping constant.
        sigma: Sigma parameter of Euler-Maruyama integrator.
        alpha: Alpha parameter of Categorical reference.
        lambda_num: Weight of the loss for continuous features.
        lambda_cat: Weight of the loss for categorical features.
        ce_lambda: Scaling weight for the auxiliary cross-entropy inside CSBM.
        num_ref_mean: Mean value of the numerical reference process.
        num_ref_std: Standard deviation of the numerical reference process.
        noise: Euler-Maruyama noise parameter.
        lr: Learning rate.
        batch_size: Batch size.
        epochs_per_direction: How many epochs to train model on each direction.
        grad_clip: Gradient clipping parameter.
        device: Device to run the model on.
        cat_dtype: torch.dtype of categorical reference process.
        num_dtype: torch.dtype of continuous reference process.
        seed: Random seed.
    """
    fb_sequence: Tuple[FB, ...] = ("b", "f", "b", "f", "b")

    cat_emb_dim: int = 16
    hidden_dim: int = 512
    time_dim: int = 128
    n_layers: int = 5
    dropout: float = 0.1
    weight_decay: float = 1e-2

    num_steps: int = 100
    eps: float = 1e-4
    sigma: float = 0.1
    alpha: float = 0.01
    lambda_num: float = 0.8
    lambda_cat: float = 0.2
    ce_lambda: float = 0.001
    num_ref_mean: float = 0.0
    num_ref_std: float = 1.0
    noise: bool = True

    lr: float = 1e-4
    batch_size: int = 256
    epochs_per_direction: int = 5
    grad_clip: Optional[float] = 1.0

    device: str = "cpu"
    cat_dtype: torch.dtype = torch.float32
    num_dtype: torch.dtype = torch.float32
    seed: int = 42
"""
from sbtab.baselines.ctgan import CTGANWrapper

# Внутри цикла фолдов:
model = CTGANWrapper(epochs=300, pac=10)
model.fit(D_train_norm)
X_syn_norm = model.sample(len(D_train_norm))
"""

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN
from typing import Optional, Union, Tuple

class CTGANWrapper:
    """
    Адаптер для модели CTGAN, интегрированный в экосистему sbtab.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        generator_dim: Tuple[int, int] = (512, 512),
        discriminator_dim: Tuple[int, int] = (512, 512),
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        batch_size: int = 500,
        epochs: int = 300,
        pac: int = 10,
        cuda: bool = True,
        seed: int = 42,
        verbose: bool = False
    ):
        # Сохраняем параметры для воспроизводимости
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = CTGAN(
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
            batch_size=batch_size,
            epochs=epochs,
            pac=pac,
            verbose=verbose,
            cuda=cuda if torch.cuda.is_available() else False
        )
        self._fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Обучение модели. 
        X: матрица признаков (предполагается, что данные уже нормализованы)
        """
        if isinstance(X, np.ndarray):
            # Конвертируем в DataFrame, так как CTGAN работает только с ними
            cols = [f"f{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
        else:
            df = X

        # В нашем проекте мы используем continuous-only данные, 
        # поэтому discrete_columns всегда пустой список.
        self.model.fit(df, discrete_columns=[])
        self._fitted = True
        return self

    def sample(self, n: int) -> np.ndarray:
        """
        Генерация синтетических данных.
        n: количество строк
        """
        if not self._fitted:
            raise RuntimeError("Модель CTGAN должна быть обучена перед сэмплированием!")

        # Сэмплируем DataFrame и переводим обратно в numpy float32
        syn_df = self.model.sample(n)
        return syn_df.to_numpy().astype(np.float32)
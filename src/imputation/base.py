from __future__ import annotations

import numpy as np
import pandas as pd


class BaseImputer:
    """
    Basisklasse für alle Imputer-Methoden.
    """

    def __init__(self, name: str) -> None:
        """Initialize the object state."""
        self.name = name

    def fit(self, X: pd.DataFrame | np.ndarray) -> BaseImputer:
        """
        Trainiert das Imputer-Modell auf den Daten X.
        """
        raise NotImplementedError("fit() needs to be implemented in subclass.")

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """
        Imputiert die fehlenden Werte in X.
        """
        raise NotImplementedError("transform() need to me implemented in subclass.")

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """
        Kombiniert fit und transform.
        """
        self.fit(X)
        return self.transform(X)

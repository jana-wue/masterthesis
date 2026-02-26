class BaseImputer:
    """
    Basisklasse für alle Imputer-Methoden.
    """

    def __init__(self, name: str):
        self.name = name
    def fit(self, X):
        """
        Trainiert das Imputer-Modell auf den Daten X.
        """
        raise NotImplementedError("fit() needs to be implemented in subclass.")

    def transform(self, X):
        """
        Imputiert die fehlenden Werte in X.
        """
        raise NotImplementedError("transform() need to me implemented in subclass.")

    def fit_transform(self, X):
        """
        Kombiniert fit und transform.
        """
        self.fit(X)
        return self.transform(X)

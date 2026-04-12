from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Possible target cols (coming from the visitation file)
TARGET_COLS = ["NACCMMSE", "CDRSUM", "MEMORY", "CDRLANG", "DEMENTED"]

def clean_features(
    df: pd.DataFrame,
    target_col=None,
    extra_drop_cols: Optional[Sequence[str]] = None
) -> pd.DataFrame:

    df = df.copy()

    drop_cols = [
        # standarize cols
        "ID", "DATE",
        # from imaging data
        "NACCID", "NACCADC", "LONIUID", "LONIUID_MULTI",
        "SCANDATE", "PROCESSDATE", "TRACER", "ACQUISITION_TIME",
        # extra ones from visit data
        "VISITYR", "VISITMO", "VISITDAY",
    ]
    
    if target_col is not None:
        drop_cols.append(target_col)

    # drop ALL targets (including the current one)
    drop_cols.extend(TARGET_COLS)

    # Optional extras
    if extra_drop_cols is not None:
        drop_cols.extend(extra_drop_cols)

    # Drop the known feature cols
    df = df.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric
    df = df.select_dtypes(include=["number"])

    return df


class PCATransformer:
    """
    Fit imputer + scaler + PCA on train data, then reuse on val/test data.
    """

    def __init__(self, n_components=30):
        # Handle the PCA as a int or ratio
        if isinstance(n_components, float) and n_components >= 1:
            if n_components.is_integer():
                n_components = int(n_components)
                
        self.n_components = n_components
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)

        self.feature_names_: Optional[list[str]] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessing and PCA on training data.
        Returns transformed train matrix.
        """

        self.feature_names_ = X.columns.tolist()
        print(f"Number of features before PCA: {len(self.feature_names_)}")
        
        # PCA
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        X_pca = self.pca.fit_transform(X_scaled)
        
        self.is_fitted = True

        print(f"PCA fitted with n_components={self.n_components}")
        print("Original shape:", X.shape)
        print("Transformed shape:", X_pca.shape)
        print("Explained variance ratio sum:", self.pca.explained_variance_ratio_.sum())

        return X_pca

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply fit imputer + scaler + PCA to new data.
        """
        if not self.is_fitted:
            raise ValueError("PCATransformer must be fit before calling transform().")

        # enforce same columns/order as training set
        missing_cols = [col for col in self.feature_names_ if col not in X.columns]
        extra_cols = [col for col in X.columns if col not in self.feature_names_]

        # Make sure all the cols are there
        if missing_cols:
            raise ValueError(f"Missing columns at transform time: {missing_cols}")

        X = X[self.feature_names_]

        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        X_pca = self.pca.transform(X_scaled)

        return X_pca

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X)


def prepare_pca_features(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    extra_drop_cols: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    From merged dataframe -> numeric feature dataframe for PCA.
    """
    X = clean_features(
        df=df,
        target_col=target_col,
        extra_drop_cols=extra_drop_cols
    )
    return X
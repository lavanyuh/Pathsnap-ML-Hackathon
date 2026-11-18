"""Train-only preprocessing (median imputation + standardisation). Fit on TRAIN trips ONLY, then reuse everywhere."""
from pathlib import Path
import numpy as np, joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .columns import FEATURE_COLS

DEFAULT_PATH = Path("artifacts/preprocess.joblib")


def fit_preprocessor(train_features_df, path=DEFAULT_PATH):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    pipe.fit(train_features_df[FEATURE_COLS].to_numpy(np.float32))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_cols": FEATURE_COLS}, path)
    return pipe


def load_preprocessor(path=DEFAULT_PATH):
    return joblib.load(path)["pipeline"]


def transform(df, pipe=None):
    """Scaled float32 matrix for models that need scaling (LogReg, KNN, GaussianNB, MLP). Trees can use raw columns."""
    pipe = pipe or load_preprocessor()
    return pipe.transform(df[FEATURE_COLS].to_numpy(np.float32)).astype(np.float32)

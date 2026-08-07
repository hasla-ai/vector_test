import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_uci(dataset_id=601):
    """UCI 로드를 시도하고, 실패 시 빠른 대체 센서 데이터를 생성합니다."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=dataset_id)
        X, y = ds.data.features.copy(), ds.data.targets.copy()
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]
        return X, y
    except Exception as e:
        from sklearn.datasets import make_regression
        Xa, ya = make_regression(
            n_samples=2000,
            n_features=5,
            noise=10,
            random_state=RANDOM_STATE
        )
        cols = ['air_temp', 'process_temp', 'rot_speed', 'torque', 'tool_wear']
        return pd.DataFrame(Xa, columns=cols), pd.Series(ya)


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


X_raw, y_raw = load_uci(601)
X_df = numeric_frame(X_raw).iloc[:, :5]     # 센서 5개
A_small = X_df.iloc[:24].values             # 24개 샘플 (4 x 6)
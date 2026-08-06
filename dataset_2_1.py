import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_uci(dataset_id):
    """UCI에서 데이터를 불러오고, 실패하면 대체 데이터를 사용합니다."""
    try:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=dataset_id)
        X, y = ds.data.features.copy(), ds.data.targets.copy()
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]
        return X, y
    except Exception as e:
        print("[안내] UCI 로드 실패:", e)
        print("[안내] 대체 데이터(sklearn breast cancer)로 진행합니다.")
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer(as_frame=True)
        return data.data, data.target


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include="number").copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


def plot_pair(before, after, label_after, title):
    """변환 전후를 같은 축에서 비교합니다."""
    plt.figure(figsize=(6, 6))
    plt.scatter(before[:, 0], before[:, 1], s=10, alpha=0.5, label="original")
    plt.scatter(after[:, 0], after[:, 1], s=10, alpha=0.5, label=label_after)
    plt.axhline(0, lw=0.8, color="gray")
    plt.axvline(0, lw=0.8, color="gray")
    plt.gca().set_aspect("equal")  # 축 비율을 맞춰야 왜곡되지 않음
    plt.legend()
    plt.title(title)
    plt.show()


def apply_T(M, X):
    """변환을 '열벡터 관점'으로 적용합니다. (X @ M.T)"""
    return np.asarray(X) @ np.asarray(M).T


# 좌표평면 데이터 전처리 (특성 2개 추출 및 표준화)
X_raw, y_raw = load_uci(17)
X2 = StandardScaler().fit_transform(numeric_frame(X_raw).iloc[:, :2])
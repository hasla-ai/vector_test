import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_uci(dataset_id=17):
    """UCI에서 Breast Cancer Wisconsin 데이터를 불러오고, 네트워크 오류 시 sklearn 로컬 데이터로 전환합니다."""
    n_retry = 3
    for attempt in range(1, n_retry + 1):
        try:
            from ucimlrepo import fetch_ucirepo

            ds = fetch_ucirepo(id=dataset_id)
            X, y = ds.data.features.copy(), ds.data.targets.copy()
            if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
                y = y.iloc[:, 0]
            return X, y
        except Exception as e:
            print(f"[안내] UCI 로드 시도 ({attempt}/{n_retry}회 실패): {e}")
            if attempt < n_retry:
                time.sleep(2 * attempt)

    print("=" * 72)
    print(
        "[안내] UCI 로드 최종 실패로 대체 데이터(sklearn breast cancer)로 진행합니다."
    )
    print("=" * 72)
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    return data.data, data.target


def numeric_frame(X):
    """수치형 컬럼만 추출하고 결측값을 중앙값으로 대체합니다."""
    Xn = X.select_dtypes(include="number").copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


# 전역 데이터 로드 및 정제/표준화
X_raw, y_raw = load_uci(17)
X_clean = numeric_frame(X_raw)

scaler = StandardScaler()
Xs = scaler.fit_transform(X_clean)

# 공분산 행렬 (rowvar=False : 각 열이 변수)
cov = np.cov(Xs, rowvar=False)

if __name__ == "__main__":
    print("표준화 데이터 shape :", Xs.shape)
    print("공분산 행렬 shape  :", cov.shape)
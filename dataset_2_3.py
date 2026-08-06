import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_uci(dataset_id=186):
    """UCI에서 데이터를 불러오고, 네트워크 오류 시 로컬 대체 데이터(Wine)로 자동 전환합니다."""
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
        "[안내] UCI 로드 최종 실패로 대체 데이터(sklearn Wine)로 진행합니다."
    )
    print("=" * 72)
    from sklearn.datasets import load_wine

    data = load_wine(as_frame=True)
    return data.data, data.target


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include="number").copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


def describe_solution(A, b):
    """Rank 비교를 통해 연립방정식 Ax = b의 해의 종류(유일해, 무한해, 불능)를 판별합니다."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))
    n_vars = A.shape[1]

    if rank_A < rank_Ab:
        kind = "해 없음(불능)"
    elif rank_A == n_vars:
        kind = "유일해"
    else:
        kind = "무한해"

    return {
        "rank(A)": rank_A,
        "rank([A|b])": rank_Ab,
        "변수 수": n_vars,
        "판정": kind,
    }


# 전역 데이터 정의 및 스케일링
X_raw, y_raw = load_uci(186)  # Wine Quality (id=186)
X_df = numeric_frame(X_raw)
A_wine = StandardScaler().fit_transform(X_df)

if __name__ == "__main__":
    print("Wine Quality 데이터 shape:", X_df.shape)
    print("컬럼 목록:", list(X_df.columns[:5]))

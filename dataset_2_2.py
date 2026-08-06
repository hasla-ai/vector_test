import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_uci(dataset_id):
    """UCI에서 데이터를 불러오고, 실패하면 대체 데이터를 사용합니다."""
    n_retry, last_err = 4, None
    for attempt in range(1, n_retry + 1):
        try:
            from ucimlrepo import fetch_ucirepo

            ds = fetch_ucirepo(id=dataset_id)
            X, y = ds.data.features.copy(), ds.data.targets.copy()
            if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
                y = y.iloc[:, 0]
            return X, y
        except Exception as e:
            last_err = e
            print(f"[안내] UCI 로드 실패 ({attempt}/{n_retry}회):", e)
            if attempt < n_retry:
                time.sleep(2 * attempt)

    if last_err is not None:
        print("=" * 72)
        print(
            "[경고] UCI 로드가 최종 실패해 대체 데이터(make_regression)로 진행합니다."
        )
        print(
            "[경고] 대체 데이터는 컬럼이 서로 독립으로 생성되어 상관계수가 거의 0입니다."
        )
        print(
            '[경고] 따라서 문제 2-1의 "상관 0.876인데도 rank가 5"라는 실데이터 해석은'
        )
        print(
            "[경고] 이 대체 데이터로는 재현되지 않습니다. 상관계수 숫자 자체가 아니라"
        )
        print(
            '[경고] "상관이 높은 것과 rank가 깎이는 것은 별개"라는 논리에만 집중하세요.'
        )
        print("=" * 72)
        print("[최종 오류]", last_err)
        from sklearn.datasets import make_regression

        Xa, ya = make_regression(
            n_samples=2000,
            n_features=5,
            noise=10,
            random_state=RANDOM_STATE,
        )
        cols = [
            "air_temp",
            "process_temp",
            "rot_speed",
            "torque",
            "tool_wear",
        ]
        return pd.DataFrame(Xa, columns=cols), pd.Series(ya)


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include="number").copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


X_raw, y_raw = load_uci(601)  # AI4I 2020 Predictive Maintenance
X_df = numeric_frame(X_raw)
A = StandardScaler().fit_transform(X_df)
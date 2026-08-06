import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_uci(dataset_id):
    """UCI에서 데이터를 불러오고, 실패하면 구조가 비슷한 대체 데이터를 사용합니다."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=dataset_id)
        X, y = ds.data.features.copy(), ds.data.targets.copy()
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]
        return X, y
    except Exception as e:
        print('[안내] UCI 로드 실패:', e)
        print('[안내] 대체 데이터로 진행합니다. shape 규칙과 해석은 동일합니다.')
        from sklearn.datasets import make_classification
        Xa, ya = make_classification(n_samples=3000, n_features=12,
                                     n_informative=6, random_state=RANDOM_STATE)
        cols = [f'feature_{i}' for i in range(Xa.shape[1])]
        return pd.DataFrame(Xa, columns=cols), pd.Series(ya)


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


X_raw, y_raw = load_uci(222)   # Bank Marketing
Xn = numeric_frame(X_raw)

FEATURE_COLS = ['age', 'balance', 'duration', 'pdays', 'previous']   # 값이 실제로 흩어져 있는(분산 있는) 컬럼만 명시적으로 선택
FEATURE_COLS = [c for c in FEATURE_COLS if c in Xn.columns] or list(Xn.columns[:5])

X_small = Xn[FEATURE_COLS].sample(n=8, random_state=RANDOM_STATE)   # 앞 8행 대신 무작위 8명 샘플링
X = StandardScaler().fit_transform(X_small)      # 스케일을 맞춰 값 비교를 쉽게
print('데이터 행렬 X:', X.shape)
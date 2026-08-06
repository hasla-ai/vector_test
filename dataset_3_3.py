import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_uci(dataset_id=186):
    """
    UCI 데이터를 로컬 캐시 폴더(./data)에 CSV로 저장하거나,
    실패할 경우 sklearn wine 데이터셋으로 대체합니다.
    """
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    # pyarrow 필요 없는 csv 포맷으로 변경
    cache_path = os.path.join(data_dir, f"uci_dataset_{dataset_id}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        X = df.drop(columns=['target'])
        y = df['target']
        return X, y

    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=dataset_id)
        X, y = ds.data.features.copy(), ds.data.targets.copy()
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]

        df_to_save = X.copy()
        df_to_save['target'] = y
        df_to_save.to_csv(cache_path, index=False)
        return X, y
    except Exception as e:
        print('[안내] UCI 로드 실패:', e)
        print('[안내] 대체 데이터(sklearn wine)로 진행합니다.')
        from sklearn.datasets import load_wine
        data = load_wine(as_frame=True)
        return data.data, data.target


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


# 데이터 전처리 및 설계행렬 구성
X_raw, y_raw = load_uci(186)                       # Wine Quality
Xn = numeric_frame(X_raw).iloc[:, :5]              # 특성 5개
y = pd.to_numeric(y_raw).values.astype(float)

Xs = StandardScaler().fit_transform(Xn)
Xd = np.column_stack([np.ones(len(Xs)), Xs])        # 절편 컬럼 + 특성 5개
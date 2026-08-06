
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_uci(dataset_id=186, n_retry=3, wait_sec=2.0):
    """UCI에서 Wine Quality 데이터를 불러옵니다. 실패 시 sklearn load_wine으로 대체합니다."""
    for attempt in range(1, n_retry + 1):
        try:
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=dataset_id)
            X, y = ds.data.features.copy(), ds.data.targets.copy()
            if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
                y = y.iloc[:, 0]
            return X, y, 'uci'
        except Exception as e:
            print(f'[안내] UCI 로드 실패 ({attempt}/{n_retry}):', e)
            if attempt < n_retry:
                print(f'[안내] {wait_sec}초 후 재시도합니다.')
                time.sleep(wait_sec)

    print('=' * 78)
    print('[경고] UCI Wine Quality 로드에 최종 실패해 대체 데이터(sklearn load_wine)로 진행합니다.')
    print('[경고] load_wine은 특성 수(13개)와 샘플 수가 모두 다른 "별개의" 데이터셋입니다.')
    print('[경고] 따라서 이후 모든 문제의 정답 수치가 교안에 적힌 값과 다르게 나옵니다.')
    print('=' * 78)
    from sklearn.datasets import load_wine
    data = load_wine(as_frame=True)
    return data.data, data.target, 'fallback'


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


# 데이터 로드 및 표준화
X_raw, y_raw, DATA_SOURCE = load_uci(186)
X_df = numeric_frame(X_raw)
Xs = StandardScaler().fit_transform(X_df)

if __name__ == '__main__':
    print('데이터 출처:', DATA_SOURCE)
    print('표준화 데이터 shape:', Xs.shape, '/ 특성 수:', X_df.shape[1])
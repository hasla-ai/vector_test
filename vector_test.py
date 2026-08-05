# 최초 1회만 실행 (새 환경일 때)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def _fallback(reason):
    """UCI 데이터를 쓸 수 없을 때 대체 데이터셋을 반환합니다."""
    print(f'[경고] {reason}')
    print('[경고] UCI Wine Quality를 불러오지 못해 다른 데이터셋(sklearn wine)으로 대체됩니다.')
    print('[경고] 따라서 아래 출력값은 교안의 예시 값과 다르게 나옵니다. 계산·해석 방법은 동일합니다.')
    from sklearn.datasets import load_wine
    data = load_wine(as_frame=True)
    return data.data, data.target


def load_uci(dataset_id):
    """UCI에서 데이터를 불러오고, 실패 원인(설치/네트워크)을 구분해 안내합니다."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        return _fallback('ucimlrepo 패키지가 설치되어 있지 않습니다. uv add ucimlrepo를 먼저 실행하세요.')

    try:
        ds = fetch_ucirepo(id=dataset_id)
    except Exception as e:
        return _fallback(f'UCI 서버 접속에 실패했습니다(네트워크·방화벽 확인 필요): {e}')

    X, y = ds.data.features.copy(), ds.data.targets.copy()
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]  # 컬럼 1개짜리 DataFrame -> Series
    return X, y


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


if __name__ == "__main__":
    # Wine Quality (ID: 186) 로드 및 전처리
    X_raw, y_raw = load_uci(186)
    X = numeric_frame(X_raw)

    print("=== 데이터 로드 결과 ===")
    print('데이터 shape:', X.shape)
    print('\n수치형 특성(Feature) 목록:')
    print(list(X.columns))
    print('\n상위 3개 행 데이터:')
    print(X.head(3))
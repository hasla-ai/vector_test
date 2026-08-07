# dataset_4_3.py
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# [수정] 기존 코드를 다른 파일에서 재사용할 수 있도록 load_bank_data() 함수로 구조화
def load_bank_data(dataset_id=222):
    """UCI Bank Marketing 데이터를 불러오거나 실패 시 대체 데이터를 생성합니다."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=dataset_id)
        X_raw, y_raw = ds.data.features.copy(), ds.data.targets.copy()
        if isinstance(y_raw, pd.DataFrame) and y_raw.shape[1] == 1:
            y_raw = y_raw.iloc[:, 0]
    except Exception as e:
        print('[안내] UCI 로드 실패:', e)
        print('[안내] 대체 데이터로 진행합니다. 분석 흐름은 동일합니다.')
        from sklearn.datasets import make_classification
        Xa, ya = make_classification(n_samples=4000, n_features=10,
                                     n_informative=5, random_state=RANDOM_STATE)
        cols = [f'feature_{i}' for i in range(Xa.shape[1])]
        X_raw, y_raw = pd.DataFrame(Xa, columns=cols), pd.Series(ya)

    # [수정] 수치형 컬럼 추출 및 결측값 보정 로직을 내부에 통합
    X_df = X_raw.select_dtypes(include='number').copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    y = pd.Series(y_raw)
    # [수정] pandas 버전 호환 타깃 매핑 분기 유지

# 주의: 최신 pandas는 문자열 컬럼의 기본 dtype이 object가 아니라 str이다.
# 그래서 `y.dtype == 'object'`로 분기하면 매핑이 건너뛰어지고,
# 바로 아랫줄 astype(int)에서 ValueError가 난다.
# pandas 버전에 상관없이 동작하도록 '숫자가 아니면 매핑'으로 판단한다.

    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
        if y.isna().any():
            raise ValueError(f'매핑되지 않은 타깃 값 {int(y.isna().sum())}건이 있습니다.')
    y = y.astype(int).values

    return X_df, y


# [수정] 모듈 직접 실행 시 데이터 구조 확인
if __name__ == '__main__':
    X_df, y = load_bank_data()
    print('수치형 특성:', X_df.shape, '/ 타깃 분포:', np.bincount(y))
import os
import platform
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# dataset_3_3 모듈에서 데이터 불러오기
from dataset_3_3 import Xd, y, Xs, Xn

# Matplotlib/폰트 관련 경고 로그 제어
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# =========================================================
# [문제 1-1] 내적으로 직교 여부 판별하기
# =========================================================
def run_problem_1_1():
    print("=" * 60)
    print("[문제 1-1] 내적으로 직교 여부 판별하기")
    print("=" * 60)

    a, b = np.array([1, 0]), np.array([0, 1])
    c, d = np.array([1, 2]), np.array([2, -1])
    e, f = np.array([1, 2]), np.array([2, 3])

    dot_ab = np.dot(a, b)
    dot_cd = np.dot(c, d)
    dot_ef = np.dot(e, f)

    # 코사인 유사도 및 사잇각(도) 계산
    cos_theta = dot_ef / (np.linalg.norm(e) * np.linalg.norm(f))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    df_res = pd.DataFrame({
        '벡터 쌍': ['(a, b)', '(c, d)', '(e, f)'],
        '내적 값': [dot_ab, dot_cd, dot_ef],
        '직교 여부': [np.isclose(dot_ab, 0), np.isclose(dot_cd, 0), np.isclose(dot_ef, 0)]
    })

    print(df_res.to_string(index=False))
    print(f"\n• 직교하지 않는 쌍 (e, f)의 사잇각: {angle_deg:.2f}도")

    meaning = "직교하는 벡터들은 성분 간 상관성이 0이므로 한 벡터가 담고 있는 정보가 다른 벡터에 전혀 겹치지 않습니다."
    print(f"\n• 직교의 의미: {meaning}\n")

if __name__ == "__main__":
    run_problem_1_1()
        

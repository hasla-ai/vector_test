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

# =========================================================
# [문제 1-2] 직교행렬 QᵀQ = I 확인하기
# =========================================================
def run_problem_1_2():
    print("=" * 60)
    print("[문제 1-2] 직교행렬 QᵀQ = I 확인하기")
    print("=" * 60)

    Q, R = np.linalg.qr(Xd)

    print(f"1. Q shape: {Q.shape}, R shape: {R.shape}")

    QtQ = Q.T @ Q
    identity_diff = np.max(np.abs(QtQ - np.eye(Q.shape[1])))
    print(f"2. QᵀQ가 단위행렬(I)과의 최대 오차: {identity_diff:.2e} (≈ 0, QᵀQ = I 성립)")

    col_norms = np.linalg.norm(Q, axis=0)
    print(f"3. Q의 열 벡터 길이 (모두 1): {np.round(col_norms, 5)}")

    v = np.random.randn(Q.shape[1])
    v_rec = Q.T @ (Q @ v)
    print(f"4. Qᵀ(Qv)와 v의 차이 노름: {np.linalg.norm(v_rec - v):.2e} (Qᵀ가 왼쪽 역행렬 역할 수행)")

    QQt = Q @ Q.T
    print(f"   QQᵀ가 단위행렬(I)인지 확인: {np.allclose(QQt, np.eye(Q.shape[0]))} (직사각 행렬이므로 단위행렬이 아님)")

    meaning = "역행렬을 복잡한 연산으로 따로 구하지 않고 전치(Qᵀ)만으로 원래 좌표를 되돌릴 수 있어 계산이 빠르고 수치적으로 안정적입니다."
    print(f"\n5. 직교행렬의 장점: {meaning}\n")


# =========================================================
# [문제 2-1] 정규방정식으로 회귀계수 계산하기
# =========================================================
def run_problem_2_1():
    print("=" * 60)
    print("[문제 2-1] 정규방정식으로 회귀계수 계산하기")
    print("=" * 60)

    # 1. 정규방정식 공식: x̂ = (XᵀX)⁻¹Xᵀy
    coef_formula = np.linalg.inv(Xd.T @ Xd) @ Xd.T @ y

    # 2. np.linalg.lstsq
    coef_lstsq, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)

    # 3. sklearn LinearRegression (fit_intercept=False)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xd, y)
    coef_sklearn = lr.coef_

    # 노름 차이 비교
    diff_1 = np.linalg.norm(coef_formula - coef_lstsq)
    diff_2 = np.linalg.norm(coef_formula - coef_sklearn)

    # RMSE 계산
    rmse_formula = np.sqrt(mean_squared_error(y, Xd @ coef_formula))
    rmse_lstsq = np.sqrt(mean_squared_error(y, Xd @ coef_lstsq))
    rmse_sklearn = np.sqrt(mean_squared_error(y, Xd @ coef_sklearn))

    df_coef = pd.DataFrame({
        '정규방정식': coef_formula,
        'lstsq': coef_lstsq,
        'sklearn': coef_sklearn
    })
    print("1. 세 방식의 회귀계수 비교:")
    print(np.round(df_coef, 5))

    print(f"\n2. 계수 간 차이(노름):")
    print(f"   • 정규방정식 vs lstsq: {diff_1:.2e}")
    print(f"   • 정규방정식 vs sklearn: {diff_2:.2e}")

    df_rmse = pd.DataFrame({
        '방식': ['정규방정식', 'np.linalg.lstsq', 'sklearn LinearRegression'],
        'RMSE': [rmse_formula, rmse_lstsq, rmse_sklearn]
    })
    print("\n3. 세 방식의 RMSE 비교 표:")
    print(df_rmse.to_string(index=False))

    meaning = "세 방식 모두 오차 제곱합을 최소화하는 동일한 수학적 대상(최소제곱 문제)을 풀기 때문에 회귀계수와 RMSE가 사실상 동일합니다."
    print(f"\n4. 결과가 동일한 이유: {meaning}\n")

# =========================================================
# [문제 2-2] 잔차의 직교성으로 투영 의미 확인하기
# =========================================================
def run_problem_2_2():
    print("=" * 60)
    print("[문제 2-2] 잔차의 직교성으로 투영 의미 확인하기")
    print("=" * 60)

    coef_lstsq, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ coef_lstsq
    resid = y - pred

    norm_xt_r = np.linalg.norm(Xd.T @ resid)
    print(f"1. Xᵀ · 잔차 의 노름: {norm_xt_r:.2e} (≈ 0)")

    print("\n2. 설계행렬의 컬럼별 잔차와의 내적 값:")
    for idx in range(Xd.shape[1]):
        dot_val = np.dot(Xd[:, idx], resid)
        print(f"   • 컬럼 {idx}: {dot_val:.2e}")

    # 계수를 임의로 변경한 경우 비교
    coef_bad = coef_lstsq.copy()
    coef_bad[1] += 0.5
    resid_bad = y - (Xd @ coef_bad)

    norm_xt_r_bad = np.linalg.norm(Xd.T @ resid_bad)
    sse_opt = np.sum(resid ** 2)
    sse_bad = np.sum(resid_bad ** 2)

    df_comp = pd.DataFrame({
        '지표': ['Xᵀ · 잔차 노름', '오차 제곱합 (SSE)'],
        '최적 계수': [norm_xt_r, sse_opt],
        '임의 계수': [norm_xt_r_bad, sse_bad]
    })
    print("\n3. 최적 계수 vs 임의 계수 비교:")
    print(df_comp.to_string(index=False))

    meaning = (
        "설계행렬의 열공간 안에서 관측값 y에 가장 가까운 점은 수직으로 내린 투영점(Xx)입니다.\n"
        "이 때 최단거리를 이루는 잔차 벡터는 열공간 내부의 모든 축과 직교하게 되며(Xᵀr = 0),\n"
        "이 직교 조건이 오차 제곱합을 최소로 만드는 최적 근사해의 기하학적 성질입니다."
    )
    print(f"\n4. 투영 관점의 해석:\n{meaning}\n")

if __name__ == "__main__":
    run_problem_2_2()


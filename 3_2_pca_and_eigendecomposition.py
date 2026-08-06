
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv, det, matrix_rank, cond
from sklearn.decomposition import PCA

from dataset_3_2 import X_df, Xs, y_raw, DATA_SOURCE

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =========================================================
# [문제 1-1] A = PDP⁻¹ 구성하고 복원하기
# =========================================================
def run_problem_1_1():
    print("=== [문제 1-1] A = PDP⁻¹ 구성하고 복원하기 ===")
    A = np.array([[4.0, 2.0], [1.0, 3.0]])

    # 1. 고유분해
    evals, evecs = np.linalg.eig(A)
    P = evecs
    D = np.diag(evals)

    print("1. P 및 D 행렬")
    print("• P (고유벡터 행렬):\n", P)
    print("• D (고유값 대각행렬):\n", D)

    # 2. 복원 및 검증
    A_reconstructed = P @ D @ inv(P)
    is_same = np.allclose(A, A_reconstructed)

    print("\n2. PDP⁻¹ 복원 검증")
    print("• PDP⁻¹ 계산 결과:\n", A_reconstructed)
    print("• 원본 A와 일치 여부:", is_same)

    # 3. det(P) 및 가역성 확인
    det_P = det(P)
    print(f"\n3. det(P) 값: {det_P:.6f} (≠ 0 이므로 역행렬 P⁻¹ 존재 함)")

    # 4. 설명
    print("\n4. 대각화의 의미")
    print(
        "대각화는 행렬 A를 '좌표축을 바꾸고(P⁻¹) -> 축별로 배율만 곱하고(D) -> 원래 좌표계로 되돌리는(P)' "
        "세 단계로 분해하는 것입니다.\n"
    )

if __name__ == "__main__":
    run_problem_1_1()
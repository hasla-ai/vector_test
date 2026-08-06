import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset_3_1 import Xs, cov

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =========================================================
# [문제 1-1] Av = λv 검증하기
# =========================================================
def run_problem_1_1():
    print("=== [문제 1-1] Av = λv 검증하기 ===")
    A = np.array([[4.0, 2.0], [1.0, 3.0]])

    # 1. 고유값 및 고유벡터 구하기
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("1. 고유값과 고유벡터")
    print(f"• 고유값 (λ) : {eigenvalues}")
    print(f"• 고유벡터 (v, 열 단위):\n{eigenvectors}\n")

    # 2 & 3. 첫 번째 / 두 번째 λ, v 검증
    print("2. A @ v 와 λ * v 비교 결과")
    for i in range(2):
        lmbda = eigenvalues[i]
        v = eigenvectors[:, i]
        Av = A @ v
        lv = lmbda * v
        print(f"• [λ{i+1} = {lmbda:.1f}]")
        print(f"  - A @ v{i+1} : {Av}")
        print(f"  - λ{i+1} * v{i+1} : {lv}")
        print(f"  - 일치 여부 : {np.allclose(Av, lv)}")

    # 4. 스칼라배 고유벡터 (3 * v1) 검증
    v1 = eigenvectors[:, 0]
    lmbda1 = eigenvalues[0]
    v1_3 = 3 * v1
    Av1_3 = A @ v1_3
    l_v1_3 = lmbda1 * v1_3

    print("\n3. 스칼라배 고유벡터 (3 * v1) 검증 결과")
    print(f"• A @ (3*v1)  : {Av1_3}")
    print(f"• λ1 * (3*v1) : {l_v1_3}")
    print(f"• 일치 여부   : {np.allclose(Av1_3, l_v1_3)}\n")

    # 5. 고유벡터 의미 설명
    print("4. 고유벡터의 의미 해석")
    print(
        "고유벡터에 임의의 스칼라 c를 곱해도 A(cv) = λ(cv)가 성립하므로, "
        "고유벡터는 특정 크기가 아닌 변환 후에도 유지되는 '방향(부분공간)' 자체만을 의미합니다.\n"
    )

# =========================================================
# [문제 1-2] 특성방정식과 일반 벡터 비교하기
# =========================================================
def run_problem_1_2():
    print("=== [문제 1-2] 특성방정식과 일반 벡터 비교하기 ===")
    A = np.array([[4.0, 2.0], [1.0, 3.0]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    I = np.eye(2)

    # 1. det(A - λI) 계산
    det_l1 = np.linalg.det(A - eigenvalues[0] * I)
    det_l2 = np.linalg.det(A - eigenvalues[1] * I)

    print("1. 고유값에서의 det(A - λI) 값")
    print(f"• λ1 = {eigenvalues[0]:.1f} 일 때 : {det_l1:.6e}")
    print(f"• λ2 = {eigenvalues[1]:.1f} 일 때 : {det_l2:.6e}\n")

    # 2. 고유값이 아닌 임의의 값(3.0)에서의 det(A - λI)
    non_eigen_val = 3.0
    det_non = np.linalg.det(A - non_eigen_val * I)
    print("2. 고유값이 아닌 값(3.0)에서의 det(A - λI) 값")
    print(f"• λ = 3.0 일 때 : {det_non:.6f} (0이 아님)\n")

    # 3 & 4. 일반 벡터 u=[1, 0] vs 고유벡터 v1 방향 비교
    u = np.array([1.0, 0.0])
    Au = A @ u
    u_dir = u / np.linalg.norm(u)
    Au_dir = Au / np.linalg.norm(Au)

    v1 = eigenvectors[:, 0]
    Av1 = A @ v1
    v1_dir = v1 / np.linalg.norm(v1)
    Av1_dir = Av1 / np.linalg.norm(Av1)

    print("3. 일반 벡터와 고유벡터의 방향 변화 비교")
    print(f"• 일반 벡터 u=[1, 0] 단위방향 : {u_dir} -> 변환 후 : {Au_dir} (방향 변경됨)")
    print(
        f"• 고유벡터 v1 단위방향       : {v1_dir} -> 변환 후 : {Av1_dir} (방향 유지됨)\n"
    )

    # 5. 특성방정식 설명
    print("4. 특성방정식의 의미 설명")
    print(
        "Av = λv를 (A - λI)v = 0으로 바꿨을 때 0이 아닌 영벡터가 아닌 해 v가 존재하려면 "
        "행렬 (A - λI)의 역행렬이 없어야 하므로 행렬식 det(A - λI) = 0을 만족하는 λ가 고유값이 됩니다.\n"
    )

if __name__ == "__main__":
    run_problem_1_2()
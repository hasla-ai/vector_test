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

if __name__ == "__main__":
    run_problem_1_1()
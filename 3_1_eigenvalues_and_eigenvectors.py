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

# =========================================================
# [문제 2-1] 공분산행렬 고유분해하기
# =========================================================
def run_problem_2_1():
    print("=== [문제 2-1] 공분산행렬 고유분해하기 ===")

    # 1. 대칭행렬 확인
    is_sym = np.allclose(cov, cov.T)
    print(f"1. 공분산행렬 대칭 여부 (np.allclose) : {is_sym}\n")

    # 2 & 3. eigh로 고유분해 후 내림차순 정렬
    evals, evecs = np.linalg.eigh(cov)
    sort_idx = np.argsort(evals)[::-1]
    evals_sorted = evals[sort_idx]
    evecs_sorted = evecs[:, sort_idx]

    # 4. 상위 5개 고유값 및 전체 합
    print("2. 정렬된 상위 5개 고유값 및 합계")
    print(f"• 상위 5개 고유값 : {np.round(evals_sorted[:5], 4)}")
    print(f"• 전체 고유값 합계: {np.sum(evals_sorted):.4f} (특성 수 30개와 동일)\n")

    # 5. 실수 및 비음수 여부 및 성질 설명
    is_real = np.all(np.isreal(evals_sorted))
    is_non_neg = np.all(evals_sorted >= -1e-10)

    print("3. 고유값 성질 검증")
    print(f"• 모든 고유값이 실수인가?   : {is_real}")
    print(f"• 모든 고유값이 0 이상인가? : {is_non_neg}")
    print(
        "• 성질 설명: 공분산행렬은 대칭행렬이므로 고유값이 항상 실수이며, "
        "준양정치(Positive Semi-definite) 행렬이므로 고유값이 각 축의 분산을 의미하여 항상 0 이상입니다.\n"
    )

if __name__ == "__main__":
    run_problem_2_1()
    
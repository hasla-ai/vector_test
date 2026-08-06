import numpy as np

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

    # 필수 1 : 배합 조건을 방정식으로 바꿔 한 번에 풀기
    # -------------------------------------------------------------
    # 문제 1-1 : 조건을 Ax = b로 정리하고 풀기
    # -------------------------------------------------------------

def run_problem_1_1():
    print("=== [2장-3강] 문제 1-1 출력 결과 ===")

    # 1. 연립방정식을 계수 행렬 A와 상수항 벡터 b로 표현
    #  2*x1 + 1*x2 = 8   (알코올 조건)
    #  1*x1 + 3*x2 = 13  (산도 조건)
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([8.0, 13.0])

    print("--- [1] 계수 행렬 A와 상수항 벡터 b ---")
    print(f"행렬 A:\n{A}")
    print(f"행렬 A shape : {A.shape}")
    print(f"벡터 b      : {b}")
    print(f"벡터 b shape : {b.shape}\n")

    # 2. np.linalg.solve(A, b)를 통한 해 구하기
    x = np.linalg.solve(A, b)
    x1, x2 = x[0], x[1]

    print("--- [2] 연립방정식의 해 x ---")
    print(f"해 벡터 x (x1, x2) : {x}")
    print(f"원액 A 양 (x1)     : {x1:.2f} L")
    print(f"원액 B 양 (x2)     : {x2:.2f} L\n")

    # 3. A @ x 검산 및 일치 여부 확인
    b_reconstructed = A @ x
    is_equal = np.allclose(b_reconstructed, b)

    print("--- [3] 검산 결과 (A @ x vs b) ---")
    print(f"복원된 b (A @ x)   : {b_reconstructed}")
    print(f"원래 벡터 b        : {b}")
    print(f"일치 여부          : {is_equal}\n")

    # 4. 배합 결과 해석 문장
    print("=== [4] 배합 결과 해석 ===")
    interpretation = f"원액 A {x1:.1f}L, 원액 B {x2:.1f}L를 섞으면 목표 알코올 도수(8)와 산도(13)를 정확히 맞출 수 있습니다."
    print(interpretation)

    # -------------------------------------------------------------
    # 문제 1-2 : 조건을 Ax = b로 정리하고 풀기
    # -------------------------------------------------------------

def run_problem_1_2():
    print("=== [2장-3강] 문제 1-2 출력 결과 ===")

    # [문제 1-1]의 표준 행렬 A와 벡터 b
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([8.0, 13.0])

    # 1. 역행렬(inv) 방식과 solve 방식 비교
    x_inv = np.linalg.inv(A) @ b
    x_solve = np.linalg.solve(A, b)
    diff_norm = np.linalg.norm(x_inv - x_solve)

    print("--- [1] 표준 행렬 A에서 inv vs solve 비교 ---")
    print(f"• np.linalg.inv(A) @ b 해 : {x_inv}")
    print(f"• np.linalg.solve(A, b) 해  : {x_solve}")
    print(f"• 두 해의 차이 (L2 Norm)    : {diff_norm:.6e}\n")

    # 2. 거의 선형종속에 가까운 병태 행렬(Ill-conditioned Matrix) A_ill 생성
    A_ill = np.array([[2.0, 1.0], [2.000001, 1.0000004]])
    b_ill = np.array([8.0, 8.000002])

    cond_ill = np.linalg.cond(A_ill)

    # 3. A_ill에 대해 두 방식으로 해 구하기
    x_inv_ill = np.linalg.inv(A_ill) @ b_ill
    x_solve_ill = np.linalg.solve(A_ill, b_ill)
    diff_norm_ill = np.linalg.norm(x_inv_ill - x_solve_ill)

    print("--- [2] 병태 행렬 A_ill에서 조건수 및 두 방식 차이 ---")
    print(f"• A_ill 조건수 (Condition Number) : {cond_ill:,.2f}")
    print(f"• A_ill에서 inv 해                 : {x_inv_ill}")
    print(f"• A_ill에서 solve 해               : {x_solve_ill}")
    print(f"• A_ill에서 두 해의 차이 (L2 Norm)  : {diff_norm_ill:.6e}\n")

    # 4. b_ill의 둘째 원소를 0.0000001만 미세하게 변경(Perturbation)
    b_ill_perturbed = np.array([8.0, 8.0000021])
    x_solve_perturbed = np.linalg.solve(A_ill, b_ill_perturbed)
    shift_norm = np.linalg.norm(x_solve_ill - x_solve_perturbed)

    print("--- [3] 미세한 입력을 변화(0.0000001)시켰을 때 해의 변동성 ---")
    print(f"• 변경 전 solve 해 (b_ill)           : {x_solve_ill}")
    print(f"• 미세 변경 후 solve 해 (b_perturbed) : {x_solve_perturbed}")
    print(f"• 해의 위치 이동 거리 (L2 Norm)      : {shift_norm:.6f}\n")

    # 5. 실무에서 solve 권장 이유 (한 문장)
    print("=== [4] 실무에서 inv보다 solve를 권장하는 이유 ===")
    print(
        "역행렬을 직접 구하는 연산($O(n^3)$)은 메모리 소모가 크고 부동소수점 오차 누적에 매우 취약한 반면, "
        "solve 함수는 LU 분해 기반의 고성능 수치 알고리즘을 사용하여 계산 효율성과 연산 안정성이 훨씬 뛰어나기 때문입니다."
    )

    # -------------------------------------------------------------
    # 문제 2-1 : 
    # -------------------------------------------------------------


    # -------------------------------------------------------------
    # 문제 3-1 : 
    # -------------------------------------------------------------


if __name__ == "__main__":
    run_problem_1_2()

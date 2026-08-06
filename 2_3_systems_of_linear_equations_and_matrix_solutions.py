import numpy as np

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


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


if __name__ == "__main__":
    run_problem_1_1()

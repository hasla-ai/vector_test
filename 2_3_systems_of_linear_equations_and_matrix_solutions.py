import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# dataset_2_3 모듈의 describe_solution 활용 (Fallback 지원)
try:
    from dataset_2_3 import describe_solution
except ImportError:

    def describe_solution(A, b):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        rank_A = np.linalg.matrix_rank(A)
        rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))
        n_vars = A.shape[1]
        if rank_A < rank_Ab:
            kind = "해 없음(불능)"
        elif rank_A == n_vars:
            kind = "유일해"
        else:
            kind = "무한해"
        return {
            "rank(A)": rank_A,
            "rank([A|b])": rank_Ab,
            "변수 수": n_vars,
            "판정": kind,
        }

    
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
    # 문제 2-1 : rank 비교로 해의 종류 판별하기
    # -------------------------------------------------------------

def run_problem_2_1():
    print("=== [2장-3강] 문제 2-1 출력 결과 ===")

    # 1. 세 가지 선형 시스템 정의
    systems = {
        "유일해 (System 1)": {
            "A": np.array([[2.0, 1.0], [1.0, 3.0]]),
            "b": np.array([8.0, 13.0]),
        },
        "무한해 (System 2)": {
            "A": np.array([[2.0, 1.0], [4.0, 2.0]]),
            "b": np.array([8.0, 16.0]),
        },
        "불능 (System 3)": {
            "A": np.array([[2.0, 1.0], [4.0, 2.0]]),
            "b": np.array([8.0, 20.0]),
        },
    }

    # 2. Rank 계산 및 판정 표 작성
    table_rows = []
    solve_results = {}

    for name, sys in systems.items():
        A_sys, b_sys = sys["A"], sys["b"]
        res = describe_solution(A_sys, b_sys)

        table_rows.append(
            {
                "시스템 종류": name,
                "rank(A)": res["rank(A)"],
                "rank([A|b])": res["rank([A|b])"],
                "변수 수(n)": res["변수 수"],
                "이론적 판정": res["판정"],
            }
        )

    # 3. np.linalg.solve 시도 및 예외 처리
        try:
            sol = np.linalg.solve(A_sys, b_sys)
            solve_results[name] = f"성공: x = {sol}"
        except np.linalg.LinAlgError as e:
            solve_results[name] = f"LinAlgError 발생 ({e})"

    df_summary = pd.DataFrame(table_rows)

    print("--- [1] 세 시스템의 Rank 비교 및 판정 표 ---")
    print(df_summary.to_string(index=False))
    print()

    print("--- [2] np.linalg.solve 시도 결과 및 오류 메시지 ---")
    for name, result in solve_results.items():
        print(f"• {name:<18} : {result}")
    print()

    # 4. 기하학적 해석
    print("=== [3] 기하학적 해석 (2차원 평면상 두 직선의 위치 관계) ===")
    print(
        "1) 유일해 (System 1): 두 직선의 기울기가 서로 달라 평면상에서 정확히 한 점에서 만납니다.\n"
        "2) 무한해 (System 2): 두 직선의 기울기와 y절편이 일치하여 완벽하게 겹치므로(동일 직선), 직선 위의 모든 점이 해가 됩니다.\n"
        "3) 불능   (System 3): 두 직선의 기울기는 같으나 y절편이 달라 서로 평행하여 영원히 만나지 않으므로 해가 존재하지 않습니다."
    )

    # -------------------------------------------------------------
    # 문제 3-1 : 
    # -------------------------------------------------------------

if __name__ == "__main__":
    run_problem_2_1()

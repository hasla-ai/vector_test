import numpy as np
import pandas as pd
from dataset_2_2 import A, X_df

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    # -------------------------------------------------------------
    # [문제 1-1] 목표 벡터를 선형결합으로 표현하기
    # -------------------------------------------------------------
    print("=== [2장-2강] 문제 1-1 출력 결과 ===")

    # 1. 기준 벡터 v1, v2 및 목표 벡터 target 정의
    v1 = np.array([1.0, 1.0])
    v2 = np.array([1.0, -1.0])
    target = np.array([3.0, 5.0])

    # 2. 열벡터로 행렬 V 생성: V = [v1, v2]
    V = np.column_stack([v1, v2])  # [[1, 1], [1, -1]]

    # 3. np.linalg.lstsq를 통한 계수 c1, c2 구하기 (V @ c = target)
    c, residuals, rank, s = np.linalg.lstsq(V, target, rcond=None)
    c1, c2 = c[0], c[1] 

    print("--- [1] 선형결합 계수 계산 ---")
    print(f"구한 계수 c1, c2 : c1 = {c1:.4f}, c2 = {c2:.4f}\n")

    # 4. 구한 계수로 목표 벡터 복원 확인
    reconstructed_target = c1 * v1 + c2 * v2
    is_reconstructed = np.allclose(reconstructed_target, target)

    print("--- [2] 목표 벡터 복원 검증 ---")
    print(f"복원된 벡터          : {reconstructed_target}")
    print(f"원래 목표 벡터       : {target}")
    print(f"복원 벡터 일치 여부  : {is_reconstructed}\n")

    # 5. v1, v2가 만드는 span 설명
    print("=== v1, v2가 만드는 Span의 의미 ===")
    print(
        "두 벡터 v1=[1, 1]과 v2=[1, -1]은 서로 평행하지 않은 선형독립 벡터이므로, 이 둘의 모든 선형결합으로 생성되는 Span은 2차원 실수 좌표평면 전체(R²)가 됩니다."
    )

    # -------------------------------------------------------------
    # [문제 1-2] 평행한 벡터들의 span 확인하기
    # -------------------------------------------------------------

    print("=== [2장-2강] 문제 1-2 출력 결과 ===")

    # 1. 평행한 두 벡터 w1, w2 정의 (w2 = 2 * w1)
    w1 = np.array([1.0, 2.0])
    w2 = np.array([2.0, 4.0])

    # 2. 열로 묶어 행렬 W 생성 후 Rank 계산
    W = np.column_stack([w1, w2])  # [[1, 2], [2, 4]]
    rank_W = np.linalg.matrix_rank(W)

    print("--- [1] 평행한 벡터 행렬의 Rank ---")
    print(f"행렬 W의 Rank : {rank_W}\n")

    # 3. 목표 벡터 target1 = [3, 5] 복원 시도
    target1 = np.array([3.0, 5.0])
    c1, res1, rank1, _ = np.linalg.lstsq(W, target1, rcond=None)
    recon1 = W @ c1
    error1 = np.linalg.norm(target1 - recon1)
    is_exact1 = np.allclose(recon1, target1)

    print("--- [2] target = [3, 5] (Span 외부 점) 복원 시도 ---")
    print(f"추정 계수 (c1, c2)  : {c1}")
    print(f"복원된 벡터         : {recon1}")
    print(f"목표와 일치 여부    : {is_exact1}")
    print(f"복원 오차 (L2 norm) : {error1:.4f}\n")

    # 4. 목표 벡터 target2 = [2, 4] (w1의 배수) 복원 시도
    target2 = np.array([2.0, 4.0])
    c2, res2, rank2, _ = np.linalg.lstsq(W, target2, rcond=None)
    recon2 = W @ c2
    error2 = np.linalg.norm(target2 - recon2)
    is_exact2 = np.allclose(recon2, target2)

    print("--- [3] target = [2, 4] (Span 내부 점) 복원 시도 ---")
    print(f"추정 계수 (c1, c2)  : {c2}")
    print(f"복원된 벡터         : {recon2}")
    print(f"목표와 일치 여부    : {is_exact2}")
    print(f"복원 오차 (L2 norm) : {error2:.4f}\n")

    # 5. Span이 평면이 아닌 이유 설명
    print("=== 평행한 두 벡터의 Span이 평면이 아닌 이유 ===")
    print(
        "두 벡터 w1과 w2는 동일한 방향(y = 2x 직선)을 가리키는 선형종속 관계이므로, 이들의 모든 선형결합은 2차원 평면 전체를 채우지 못하고 원점을 지나는 1차원 직선에 갇히게 됩니다."
    )

    # 필수 2 : 센서가 5개인데 진짜 정보는 몇 개일까
    # -------------------------------------------------------------
    # [문제 2-1] 데이터 행렬의 rank 계산하기
    # -------------------------------------------------------------

    print("=== [2장-2강] 문제 2-1 출력 결과 ===")

    # 1. 데이터 행렬 A의 shape 및 rank 계산
    n_samples, n_features = A.shape
    rank_A = np.linalg.matrix_rank(A)
    max_possible_rank = min(n_samples, n_features)

    print("--- [1] 데이터 행렬 A의 차원 및 Rank ---")
    print(f"행렬 A의 Shape    : {A.shape} (샘플 수: {n_samples}, 특성 수: {n_features})")
    print(f"행렬 A의 Rank     : {rank_A}")
    print(f"최대 가능 Rank    : {max_possible_rank}")
    print(
        f"Full Rank 여부    : {rank_A == max_possible_rank} (열의 개수 5개와 동일)\n"
    )

    # 2. 컬럼 간 상관계수 행렬 계산
    df_A = pd.DataFrame(A, columns=X_df.columns)
    corr_matrix = df_A.corr()

    print("--- [2] 컬럼 간 상관계수 행렬 (Correlation Matrix) ---")
    print(corr_matrix.round(4))
    print()

    # 3. 절댓값이 가장 큰 상관계수 컬럼 쌍 찾기 (자기 자신 제외)
    corr_unstack = corr_matrix.abs().unstack()
    # 대각선(1.0) 제거
    corr_unstack = corr_unstack[corr_unstack < 0.99999]
    top_pairs = corr_unstack.sort_values(ascending=False).drop_duplicates()

    print("--- [3] 절댓값 기준 상위 상관계수 컬럼 쌍 ---")
    for (c1, c2), abs_val in top_pairs.head(2).items():
        actual_val = corr_matrix.loc[c1, c2]
        print(f"• {c1} - {c2} : {actual_val:.4f} (절댓값: {abs_val:.4f})")
    print()

    # 4. 데이터 상태 및 선형독립성 해석
    print("=== [4] 데이터 상태 해석 및 선형독립성 논리 ===")
    print(
        "1) 높은 상관계수에도 Rank가 5(Full Rank)인 이유:\n"
        "   상관계수(0.876)가 매우 높다는 것은 두 센서 변수가 '강한 경향성'을 함께 보인다는 의미일 뿐, "
        "   한 컬럼이 다른 컬럼의 '정확한 상수배(c · v)'로 표현된다는 의미가 아닙니다. "
        "   미세한 노이즈나 독자적인 정보가 존재하는 한 수학적으로는 완벽한 선형독립 상태를 유지하므로 Rank 손실이 발생하지 않습니다."
    )
    print(
        "2) Rank가 실제로 깎이기 위해 필요한 상관계수:\n"
        "   한 컬럼이 다른 컬럼과 '정확하게 선형종속'이 되어 Rank가 감소하려면, "
        "   두 변수 간의 상관계수가 정확히 1.0 또는 -1.0이어야 합니다."
    )

# -------------------------------------------------------------
# [문제 2-2] 선형종속 컬럼을 추가해도 rank가 늘지 않음 확인하기
# -------------------------------------------------------------

    print("=== [2장-2강] 문제 2-2 출력 결과 ===")

    # 1. 기존 A에 선형종속 컬럼 2개 추가 -> A_dep
    col_double = (A[:, 0] * 2.0).reshape(-1, 1)  # 첫 번째 컬럼의 2배
    col_sum = (A[:, 1] + A[:, 2]).reshape(-1, 1)  # 두 번째 + 세 번째 컬럼의 합
    A_dep = np.hstack([A, col_double, col_sum])

    # 2. 기존 A에 무작위 난수 컬럼 1개 추가 -> A_rand
    rand_col = np.random.randn(A.shape[0], 1)
    A_rand = np.hstack([A, rand_col])

    # 3. Shape 및 Rank 계산
    shape_A, rank_A = A.shape, np.linalg.matrix_rank(A)
    shape_dep, rank_dep = A_dep.shape, np.linalg.matrix_rank(A_dep)
    shape_rand, rank_rand = A_rand.shape, np.linalg.matrix_rank(A_rand)

    # 4. 비교 표 출력
    print("--- [1] A / A_dep / A_rand의 Shape 및 Rank 비교 표 ---")
    print(
        f"{'행렬 종류':<10} | {'Shape (행, 열)':<16} | {'Matrix Rank':<12} | {'추가된 컬럼의 성질':<30}"
    )
    print("-" * 78)
    print(
        f"{'A (원본)':<10} | {str(shape_A):<16} | {rank_A:<12} | {'기본 센서 데이터 (5개)':<30}"
    )
    print(
        f"{'A_dep':<10} | {str(shape_dep):<16} | {rank_dep:<12} | {'선형종속 컬럼 2개 (2*col1, col2+col3)':<30}"
    )
    print(
        f"{'A_rand':<10} | {str(shape_rand):<16} | {rank_rand:<12} | {'선형독립 난수 컬럼 1개':<30}"
    )
    print("-" * 78 + "\n")

    # 5. 컬럼 추가와 Rank 변화 관계 설명
    print("=== [2] 컬럼 추가와 Rank 변화의 관계 설명 ===")
    print(
        "추가된 컬럼이 기존 컬럼들의 선형결합(스칼라배, 합 등)으로 표현 가능한 선형종속 관계이면 새로운 정보나 공간의 차원을 제공하지 못하므로 Rank가 전혀 늘어나지 않지만, "
        "기존 컬럼들로 설명할 수 없는 독자적인 차원을 가진 선형독립 컬럼(난수 등)이 추가되면 기하학적 차원이 확장되어 Rank가 1 증가합니다."
    )
# -------------------------------------------------------------
# [문제 3-1] 영공간의 비자명한 해로 중복 관계 확인하기
# -------------------------------------------------------------

    print("=== [2장-2강] 심화 1 (문제 3-1) 출력 결과 ===")

    # 문제 2-2의 A_dep 재구성 (Shape: 10000 x 7)
    # col0~col4: 원본 A, col5: 2 * col0, col6: col1 + col2
    col_double = (A[:, 0] * 2.0).reshape(-1, 1)
    col_sum = (A[:, 1] + A[:, 2]).reshape(-1, 1)
    A_dep = np.hstack([A, col_double, col_sum])

    # 1. c1 표현: 2 * col0 - col5 = 0  ->  [2, 0, 0, 0, 0, -1, 0]
    c1 = np.array([2.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0])

    # 2. A_dep @ c1 이 0 벡터에 가까운지 확인 (L2 norm)
    norm_c1 = np.linalg.norm(A_dep @ c1)

    # 3. c2 표현: col1 + col2 - col6 = 0  ->  [0, 1, 1, 0, 0, 0, -1]
    c2 = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    norm_c2 = np.linalg.norm(A_dep @ c2)

    print("--- [1] 영공간 계수 벡터 검증 (Ax = 0) ---")
    print(f"c1 벡터 (2*col0 - col5 = 0)    : {c1}")
    print(f"||A_dep @ c1|| (L2 Norm)        : {norm_c1:.6e}")
    print(f"c2 벡터 (col1 + col2 - col6 = 0) : {c2}")
    print(f"||A_dep @ c2|| (L2 Norm)        : {norm_c2:.6e}\n")

    # 4. A_dep의 컬럼 수, rank, 영공간 차원 계산 (Rank-Nullity 정리)
    n_cols_dep = A_dep.shape[1]
    rank_dep = np.linalg.matrix_rank(A_dep)
    nullity_dep = n_cols_dep - rank_dep

    # 5. A(중복 없음)의 차원 계산
    n_cols_A = A.shape[1]
    rank_A = np.linalg.matrix_rank(A)
    nullity_A = n_cols_A - rank_A

    print("--- [2] 행렬 차원 및 영공간(Nullspace) 차원 비교 ---")
    print(
        f"• A_dep : 컬럼 수 = {n_cols_dep} | Rank = {rank_dep} | 영공간 차원(Nullity) = {nullity_dep}"
    )
    print(
        f"• A     : 컬럼 수 = {n_cols_A} | Rank = {rank_A} | 영공간 차원(Nullity) = {nullity_A}\n"
    )

    # 6. 중복 특성에 대한 데이터·모델 관점의 해석
    print("=== [3] 중복 특성에 대한 데이터·모델 관점의 해석 ===")
    print(
        "영공간의 차원(Nullity)이 0보다 크다는 것은 데이터에 다른 특성들의 선형결합으로 완벽히 설명되는 'redundant(중복)' 특성이 존재함을 뜻합니다. "
        "이 경우 선형 회귀 모델에서 $(A^T A)^{-1}$ 역행렬 계산 시 유일해를 구하지 못하거나 계수(Weight)가 폭주하는 다중공선성 문제가 발생합니다. "
        "따라서 영공간을 형성하는 영벡터 $x \neq 0$를 추적함으로써 완전히 중복된 센서 특성을 찾아 제거(Feature Selection)하거나 차원을 축소해야 합니다."
    )

if __name__ == "__main__":
    main()
import numpy as np

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
    

if __name__ == "__main__":
    main()
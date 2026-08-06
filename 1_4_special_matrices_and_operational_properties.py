import numpy as np
from dataset_1_4 import X

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():

    # 필수 1 : 특수 행렬로 데이터 축을 조정하기
    # -------------------------------------------------------------
    # [문제 1-1] 단위행렬과 대각행렬 적용하기
    # -------------------------------------------------------------
    print("=== [1장-4강] 문제 1-1 출력 결과 ===")

    # 1. X의 앞 50행을 A로 추출 및 단위행렬 I 생성
    A = X[:50]  # shape: (50, 4)
    I = np.eye(A.shape[1])  # shape: (4, 4)

    # 2. A @ I == A 검증
    A_I = A @ I
    is_identity_same = np.allclose(A, A_I)
    print(f"A @ I == A 여부 (np.allclose) : {is_identity_same}\n")

    # 3. 비중 조정을 위한 대각행렬 D 생성 및 A @ D 계산
    weights = [1, 10, 0.1, 2]
    D = np.diag(weights)  # shape: (4, 4)
    A_D = A @ D  # shape: (50, 4)

    print("=== 대각행렬 확인 ===")
    print("[대각행렬]")
    print(np.round(D[:4], 4))
    
    # 4. 대각행렬 적용 전후 값 비교 (앞 3개 행 출력)
    print("=== 대각행렬 적용 전후 샘플 값 (상위 3행) ===")
    print("[A 원본 (상위 4행)]")
    print(np.round(A[:4], 4))
    print("\n[A @ D 연산 후 (상위 4행)]")
    print(np.round(A_D[:4], 4))
    print()

    # 5. 열별 표준편차 비교
    std_A = np.std(A, axis=0)
    std_A_D = np.std(A_D, axis=0)

    print("=== 열별 표준편차 변화 ===")
    for idx, (s_orig, s_diag, w) in enumerate(zip(std_A, std_A_D, weights)):
        print(
            f"특성 {idx+1} (가중치 {w:4.1f}) : 원본 std = {s_orig:.4f} -> 연산 후 std = {s_diag:.4f} (비율: {s_diag/s_orig:.1f}배)"
        )

    print("\n=== 대각행렬의 축 영향 설명 ===")
    print(
        "대각행렬 D를 우측에서 곱하면(A @ D), A의 각 열(특성 축)이 D의 대응하는 대각 원소값만큼 독립적으로 스케일링(확대/축소)됩니다."
    )

    # -------------------------------------------------------------
    # [문제 1-2] 데이터에서 대칭행렬 찾기
    # -------------------------------------------------------------

    print("=== [1장-4강] 문제 1-2 출력 결과 ===")

    # 1. G = X.T @ X 계산 및 Shape 확인
    G = X.T @ X

    print("[행렬 X] 앞 4행")
    print(np.round(X[:4], 4))

    print("[행렬 X.T]")
    print(np.round(X.T[:4], 4))
    
    print("[행렬 G = X.T @ X]")
    print(np.round(G[:4], 4))

    print("[행렬 G.T]")
    print(np.round(G.T, 4))

    print("[행렬 G.T@G]")
    print(np.round(G.T@G, 4))


    # 2. G의 대칭 여부 확인 (G == G.T)
    is_G_symmetric = np.allclose(G, G.T)

    # 3. 공분산 행렬 계산 및 대칭 여부 확인
    cov_matrix = np.cov(X, rowvar=False)
    is_cov_symmetric = np.allclose(cov_matrix, cov_matrix.T)

    # 4. 결과 출력
    print(f"G (X.T @ X) shape   : {G.shape}")
    print(f"공분산 행렬 shape   : {cov_matrix.shape}\n")

    print(f"G의 대칭 여부 (np.allclose)       : {is_G_symmetric}")
    print(f"공분산 행렬 대칭 여부 (np.allclose) : {is_cov_symmetric}\n")

    # 5. 공분산 행렬이 대칭인 이유 설명
    print("=== 공분산 행렬이 대칭인 이유 ===")
    print(
        "변수 i와 변수 j 사이의 공분산 Cov(X_i, X_j)는 변수 j와 변수 i 사이의 공분산 Cov(X_j, X_i)와 수학적으로 완벽히 동일하므로, (i, j) 위치와 (j, i) 위치의 원소가 항상 같은 대칭행렬이 됩니다."
    )

    # -------------------------------------------------------------
    # [문제 2-1] AB ≠ BA 확인하기
    # -------------------------------------------------------------
    print("=== [1장-4강] 문제 2-1 출력 결과 ===")

    # 1. 2x2 행렬 P, Q 정의
    P = np.array([[1, 2], [0, 1]])
    Q = np.array([[1, 0], [3, 1]])

    # 2. P @ Q 및 Q @ P 계산
    PQ = P @ Q
    QP = Q @ P

    print("--- [1] 일반 행렬 P, Q 곱 연산 ---")
    print(f"P @ Q :\n{PQ}")
    print(f"Q @ P :\n{QP}")
    print(
        f"P @ Q == Q @ P 동일 여부 (np.array_equal) : {np.array_equal(PQ, QP)}\n"
    )

    # 3. 특수 대각행렬과의 교환 가능 여부 테스트
    # (1) 2 * 단위행렬
    I_scalar = 2 * np.eye(2)
    commute_I_scalar = np.array_equal(I_scalar @ Q, Q @ I_scalar)

    # (2) 서로 다른 대각 원소를 가진 대각행렬 np.diag([2, 5])
    D = np.diag([2, 5])
    commute_D_Q = np.array_equal(D @ Q, Q @ D)

    # (3) 대각행렬끼리의 곱
    D2 = np.diag([3, 4])
    commute_D1_D2 = np.array_equal(D @ D2, D2 @ D1 if "D1" in locals() else D2 @ D)

    print("--- [2] 대각행렬 및 단위행렬 교환법칙 비교 ---")
    print(f"1) (2 * 단위행렬) @ Q == Q @ (2 * 단위행렬) : {commute_I_scalar}")
    print(f"2) diag([2, 5]) @ Q == Q @ diag([2, 5])       : {commute_D_Q}")
    print(f"3) 대각행렬끼리의 곱 (D1 @ D2 == D2 @ D1)      : {commute_D1_D2}\n")

    # 4. 비교환성 이유 설명
    print("=== 행렬 곱이 교환법칙을 만족하지 않는 이유 ===")
    print(
        "행렬곱 AB의 (i, j) 원소는 A의 i번째 '행'과 B의 j번째 '열'의 내적이지만, "
        "BA의 (i, j) 원소는 B의 i번째 '행'과 A의 j번째 '열'의 내적이므로 "
        "서로 전혀 다른 벡터 간의 곱셈이 수행되어 결과가 달라집니다."
    )

    # -------------------------------------------------------------
    # [문제 2-2] 전치 성질로 XᵀX가 대칭인 이유 설명하기
    # -------------------------------------------------------------
    print("=== [1장-4강] 문제 2-2 출력 결과 ===")

    # 1. 2x2 행렬 P, Q 정의
    P = np.array([[1, 2], [0, 1]])
    Q = np.array([[1, 0], [3, 1]])

    # 2. (P @ Q).T 와 Q.T @ P.T 비교
    PQ_transposed = (P @ Q).T
    QT_PT = Q.T @ P.T
    is_reverse_equal = np.array_equal(PQ_transposed, QT_PT)

    # 3. 순서를 유지한 P.T @ Q.T 비교
    PT_QT = P.T @ Q.T
    is_same_order_equal = np.array_equal(PQ_transposed, PT_QT)

    print("--- [1] 전치와 연산 순서 검증 ---")
    print(f"(P @ Q).T == Q.T @ P.T (순서 역전) 일치 여부 : {is_reverse_equal}")
    print(
        f"(P @ Q).T == P.T @ Q.T (순서 유지) 일치 여부 : {is_same_order_equal}\n"
    )

    # 4. (XᵀX)ᵀ 대칭성 검증 (수학적 전개 및 코드 확인)
    # 수식: (XᵀX)ᵀ = Xᵀ (Xᵀ)ᵀ = XᵀX
    G = X.T @ X
    G_transposed = G.T
    is_G_symmetric = np.allclose(G, G_transposed)

    print("--- [2] XᵀX의 대칭성 검증 ---")
    print(f"XᵀX shape        : {G.shape}")
    print(f"(XᵀX)ᵀ shape     : {G_transposed.shape}")
    print(f"XᵀX == (XᵀX)ᵀ 여부: {is_G_symmetric}\n")

    # 5. 공분산 행렬 대칭성 결론 설명
    print("=== 공분산 행렬이 항상 대칭인 이유 ===")
    print(
        "공분산 행렬은 중심화된 데이터 행렬 X_c에 대해 (1/n) * X_cᵀ X_c 형태를 띠는데, 전치 규칙에 의해 ((X_c)ᵀ X_c)ᵀ = (X_c)ᵀ X_c 가 되어 항상 대칭입니다."
    )    

    # -------------------------------------------------------------
    # [문제 3-1] 행렬식과 조건수로 역행렬 존재·안정성 확인하기
    # -------------------------------------------------------------
    print("=== [1장-4강] 문제 3-1 출력 결과 ===")

    # 1. 정상 행렬 A 검증
    A = np.array([[1, 2], [3, 4]], dtype=float)
    det_A = np.linalg.det(A)
    A_inv = np.linalg.inv(A)
    is_identity = np.allclose(A @ A_inv, np.eye(2))

    print("--- [1] 정상 행렬 A 검증 ---")
    print(f"det(A)                  : {det_A:.4f}")
    print(f"A @ A_inv ≈ I 여부      : {is_identity}\n")

    # 2. 특이행렬 S 검증 (행 간 다중공선성)
    S = np.array([[1, 2], [2, 4]], dtype=float)
    det_S = np.linalg.det(S)

    print("--- [2] 특이행렬 S 검증 ---")
    print(f"det(S)                  : {det_S:.4f}")
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError as e:
        print(f"발생한 오류 메시지      : LinAlgError ({e})\n")

    # 3. 실제 데이터 X와 중복 컬럼 추가 데이터 X_dup 비교
    # 첫 번째 컬럼의 2배인 컬럼을 붙임 (다중공선성 유발)
    # 3-1. X의 첫 번째 열만 슬라이싱 (shape: N x 1)
    # 3-2. 모든 값에 2를 곱함
    col0_dup = X[:, :1] * 2
    # 3-3. 기존 X(4개 열) 옆에 2배 곱한 열(1개 열)을 가로로 합침 (shape: N x 5)
    X_dup = np.hstack([X, col0_dup])  # shape: (N, 5)

    G = X.T @ X  # (4, 4)
    G_dup = X_dup.T @ X_dup  # (5, 5)

    det_G = np.linalg.det(G)
    det_G_dup = np.linalg.det(G_dup)

    cond_G = np.linalg.cond(G)
    cond_G_dup = np.linalg.cond(G_dup)

    print("--- [3] 중복 컬럼 추가 전후 $X^T X$ 비교 ---")
    print(f"원본 X^T X        - det: {det_G:.4f},  cond: {cond_G:.4f}")
    print(
        f"중복 X_dup^T X_dup - det: {det_G_dup:.4e}, cond: {cond_G_dup:.4e}\n"
    )

    # 4. 실무 대응 방안 및 이유 정리
    print("=== 실무 대응 방안 및 평가 유의점 ===")
    print(
        "1) 다중공선성 문제: 행렬식이 0에 수렴하거나 조건수가 발산(급격히 증가)하면 역행렬 연산이 불가능해지거나 수치적으로 극도로 불안정해져 가중치(회귀계수)가 폭발합니다."
    )
    print(
        "2) 지표 판단 기준: 행렬식의 절댓값은 스케일링이나 차원에 따라 수치가 왜곡될 수 있으므로, 역행렬의 수치적 안정성은 스케일 비전달 지표인 조건수(Condition Number)로 판단해야 합니다."
    )
    print(
        "3) 실무 대응: 상관관계 분석/VIF 검증을 통해 완전 중복 및 고도 상관 피처를 삭제하거나, L2 규제(Ridge: XᵀX + λI)를 적용하여 정칙화(Regularization)하거나, PCA로 차원을 축소해 해결합니다."
    )

if __name__ == "__main__":
    main()
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


if __name__ == "__main__":
    main()
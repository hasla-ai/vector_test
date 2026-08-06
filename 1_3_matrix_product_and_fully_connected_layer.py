import numpy as np
from dataset_1_3 import X

# 공통 난수 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

    # 필수 1 : 고객 수천 명의 반응 점수를 한 번에 계산하기

    # -------------------------------------------------------------
    # [문제 1-1] 데이터 행렬 X와 완전연결층 shape 확인하기
    # -------------------------------------------------------------

def main():

    # 1. 가중치 행렬 W 및 편향 벡터 b 생성 (출력 점수 3개 기준)
    W = np.random.randn(5, 3)
    b = np.random.randn(3)

    # 2. 완전연결층 연산 (Y = X @ W + b)
    Y = X @ W + b

    # 3. Shape 및 설명 출력
    print("\n=== [1장-3강] 문제 1-1 출력 결과 ===")
    print(f"X shape : {X.shape}")
    print(f"W shape : {W.shape}")
    print(f"b shape : {b.shape}")
    print(f"Y shape : {Y.shape}\n")

    print("=== 각 축의 의미 설명 ===")
    print("- X (8, 5): 고객 수 8명(Axis 0), 입력 특성 수 5개(Axis 1)")
    print("- W (5, 3): 입력 특성 수 5개(Axis 0), 출력 점수 종류 3개(Axis 1)")
    print("- b (3,)  : 출력 점수별 기본 편향값 3개")
    print("- Y (8, 3): 고객 8명의 3개 상품별 반응 점수\n")

    print("=== Y Shape 결정 근거 ===")
    print("X의 열(5)과 W의 행(5)이 일치하여 행렬곱이 가능하며,")
    print("결과 행렬 Y의 Shape은 X의 행 크기(8)와 W의 열 크기(3)에 따라 (8, 3)으로 결정됩니다.")

    # -------------------------------------------------------------
    # [문제 1-2] 반복문 계산과 행렬곱 결과 비교
    # -------------------------------------------------------------
    # 1. for 문으로 고객 한 명(행)씩 x @ W + b 계산하여 리스트에 저장
    Y_loop_list = []
    for x in X:  # x shape: (5,)
        y_i = x @ W + b  # (5,) @ (5, 3) + (3,) -> (3,)
        Y_loop_list.append(y_i)

    # 리스트를 NumPy 배열로 변환
    Y_loop = np.array(Y_loop_list)

    # 2. 결과 출력 및 비교
    
    print("\n=== [1장-3강] 문제 1-2 출력 결과 ===")
    print(f"반복문 계산 결과 shape : {Y_loop.shape}")
    print(f"두 결과의 일치 여부 (np.allclose) : {np.allclose(Y, Y_loop)}")
    print(f"두 결과의 Shape 동일 여부 : {Y.shape == Y_loop.shape}\n")

    print("=== 배치 처리의 장점 ===")
    print(
        "반복문(for) 대신 전체 데이터를 행렬(Batch) 단위로 묶어 연산하면, C 언어 수준의 벡터화(Vectorization) 및 병렬 처리가 가능해져 메모리 접근 효율과 계산 속도가 획기적으로 향상됩니다."
    )

    # -------------------------------------------------------------
    # [문제 2-1] Shape 오류 재현 및 수정
    # -------------------------------------------------------------
    print("=== [1장-3강] 문제 2-1 출력 결과 ===")

    # 1. 입력 차원과 맞지 않는 잘못된 가중치 W_wrong (4, 3) 생성
    W_wrong = np.random.randn(4, 3)

    # 2. X와 W_wrong의 Shape 나란히 출력
    print(f"X.shape: {X.shape}, W_wrong.shape: {W_wrong.shape}")

    # 3. try/except로 행렬곱 예외 처리 및 오류 메시지 출력
    try:
        Y_wrong = X @ W_wrong
    except ValueError as e:
        print(f"발생한 오류 메시지 : {e}\n")

    # 4. 올바른 Shape인 W_fixed (5, 3)로 수정하여 계산
    b = np.random.randn(3)
    W_fixed = np.random.randn(5, 3)
    Y_fixed = X @ W_fixed + b

    print(f"수정 후 계산 결과 Y_fixed shape : {Y_fixed.shape}\n")

    # 5. 행렬곱 가능 조건 정리
    print("=== 행렬곱 가능 조건 정리 ===")
    print(
        "두 행렬 A(m, n)와 B(k, l)의 행렬곱(A @ B)이 성립하려면 앞 행렬 A의 열 개수(n)와 뒤 행렬 B의 행 개수(k)가 반드시 일치해야 합니다."
    )

    # -------------------------------------------------------------
    # [문제 2-2] 전치로 shape 맞추고 전치 성질 검증하기
    # -------------------------------------------------------------

    # 가중치 행렬 W 생성 (5, 3)
    W = np.random.randn(5, 3)

    print("=== [1장-3강] 문제 2-2 출력 결과 ===")

    # 1. 전치 전후 shape 확인
    print(f"X 전치 전 shape : {X.shape}")
    print(f"X 전치 후 (X.T) shape : {X.T.shape}\n")

    # 2. (X @ W).T 와 W.T @ X.T 연산 및 일치 여부 확인
    X_W_transposed = (X @ W).T
    W_T_X_T = W.T @ X.T

    print(f"(X @ W).T shape : {X_W_transposed.shape}")
    print(f"W.T @ X.T shape  : {W_T_X_T.shape}")
    print(
        f"두 결과의 일치 여부 (np.allclose) : {np.allclose(X_W_transposed, W_T_X_T)}\n"
    )

    # 3. 순서를 유지한 경우 (X.T @ W.T) 계산 시도 및 예외 처리
    print("=== 순서를 유지한 경우 (X.T @ W.T) 시도 ===")
    try:
        invalid_product = X.T @ W.T
    except ValueError as e:
        print(f"계산 가능 여부 : 불가능 (오류 발생)")
        print(f"발생한 오류 메시지 : {e}\n")

    # 4. 전치 성질 설명
    print("=== 전치 성질 설명 ===")
    print(
        "행렬곱이 성립하려면 앞 행렬의 열과 뒤 행렬의 행 크기가 같아야 하므로, 각 행렬을 전치하면 차원의 축이 뒤바뀌어 연산 순서도 반대로 전환됩니다."
    )


if __name__ == "__main__":
    main()
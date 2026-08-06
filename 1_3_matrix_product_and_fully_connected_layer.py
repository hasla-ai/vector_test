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



if __name__ == "__main__":
    main()
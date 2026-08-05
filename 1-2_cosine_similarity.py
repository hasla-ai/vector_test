import numpy as np
import pandas as pd
from dataset2 import get_retail_data


# 공통 난수 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def main():
    # -------------------------------------------------------------
    # [데이터 로드]
    # -------------------------------------------------------------
    M_df, M = get_retail_data()

    # 필수 1 : 고객 구매 이력을 벡터로 만들고 겹치는 정도 재기

    # -------------------------------------------------------------
    # [문제 1-1] 고객 벡터의 내적 직접 계산하기
    # -------------------------------------------------------------
    # 1. M_df에서 0번, 1번 고객의 구매량 벡터 a, b 추출 및 shape 확인
    a = M_df.iloc[0].to_numpy()
    b = M_df.iloc[1].to_numpy()

    # 2. 내적 직접 계산 (성분별 곱의 합)
    dot_direct = np.sum(a * b)      

    # 3. np.dot(a, b) 및 a @ b 연산과 일치 여부 확인
    dot_np = np.dot(a, b)
    dot_matmul = a @ b

    is_equal = np.isclose(dot_direct, dot_np) and np.isclose(dot_direct, dot_matmul)

    # 4. 내적 결과의 데이터 타입 및 스칼라 확인
    result_type = type(dot_direct)

    print("=== [1장-2강] 문제 1-1 출력 결과 ===")
    print(f"1. 0번 고객 벡터 a shape: {a.shape}, 1번 고객 벡터 b shape: {b.shape}")
    print(f"2. 직접 계산한 내적 np.sum(a * b) : {dot_direct}")
    print(f"3. np.dot(a, b) 결과              : {dot_np}")
    print(f"4. a @ b 연산 결과                 : {dot_matmul}")
    print(f"5. 세 연산 방식 결과 일치 여부     : {is_equal}")
    print(f"6. 내적 결과의 반환 타입           : {result_type}\n")

    print("=== [내적 결과가 스칼라라는 점의 의미] ===")
    print("수백 개 상품에 대한 두 고객의 다차원 구매 이력을 단 하나의 수치(스칼라)로 요약함으로써, "
          "고객 간 공통 구매 성향과 유사도 규모를 즉각적으로 비교·평가할 수 있게 해줍니다.\n")


if __name__ == "__main__":
    main()
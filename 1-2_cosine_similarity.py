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

    # -------------------------------------------------------------
    # [문제 1-2] 내적 값이 구매 규모에 영향을 받는지 확인하기
    # -------------------------------------------------------------
    # 1. 0번 고객과 나머지 모든 고객의 내적 일괄 계산 (행렬-벡터 곱)
    dot_scores = M @ a  # shape: (60,)

    # 고객 ID(인덱스)와 내적 점수를 매핑한 Series 생성
    dot_series = pd.Series(dot_scores, index=M_df.index)
    sorted_dots = dot_series.sort_values(ascending=False)

    top_1_id = sorted_dots.index[0]
    top_1_score = sorted_dots.iloc[0]

    print("=== [1장-2강] 문제 1-2 출력 결과 ===")
    print(f"1. 내적 1위 고객 ID: {top_1_id} (내적 점수: {top_1_score:.2f})")
    print("   - [이유 설명]: 자기 자신과의 내적은 a · a = ||a||²로 모든 성분의 제곱합이 되므로, "
          "음수가 없는 구매량 데이터에서 수학적으로 항상 최대값을 가집니다.\n")

    # 2. 자기 자신(0번 고객)을 제외한 내적 상위 5명 선별
    top_5_except_self = sorted_dots.iloc[1:6]

    # 각 고객의 총 구매량(행 합계) 계산
    total_quantities = M_df.sum(axis=1)
    mean_quantity = total_quantities.mean()

    df_top_5 = pd.DataFrame({
        "Customer_ID": top_5_except_self.index,
        "내적 점수": top_5_except_self.values,
        "총 구매량": total_quantities.loc[top_5_except_self.index].values
    })

    print("2. 내적 상위 5명 (자기 자신 제외) 및 총 구매량:")
    print(df_top_5.to_string(index=False))

    print(f"\n3. 전체 고객의 총 구매량 평균: {mean_quantity:.2f} 개\n")

    print("=== [내적만 사용할 때의 한계 설명] ===")
    print("내적은 벡터의 방향(구매 품목 조합)뿐만 아니라 크기(총 구매량)에도 정비례하므로, "
          "실제 구매 패턴이나 취향이 유사한 고객보다 단순히 헤비 슈머(대량 구매자)가 상위로 왜곡되어 선택되는 한계가 있습니다.\n")

if __name__ == "__main__":
    main()
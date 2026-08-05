import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  # sklearn의 cosin_similarity 함수
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

    #필수 2 : 구매 규모를 제외하고 "구매 성향"만 비교하기

    # -------------------------------------------------------------
    # [문제 2-1] 코사인 유사도를 공식으로 직접 구현하기
    # -------------------------------------------------------------
    # 1. 0번과 1번 고객의 코사인 유사도 계산
    cos_a_b = cosine_similarity_single(a, b)

    # 2. 검증 케이스 계산
    cos_self = cosine_similarity_single(a, a)            # 자기 자신
    cos_scaled = cosine_similarity_single(a, 3 * a)       # 스칼라배 (3 * a)
    
    # 직교 벡터 (a와 겹치지 않는 벡터 c) 생성
    c = np.zeros_like(a)
    c[a == 0] = 1.0  # a가 0인 성분에만 1 지정
    cos_orthogonal = cosine_similarity_single(a, c)       # 직교 케이스

    print("=== [1장-2강] 문제 2-1 출력 결과 ===")
    print(f"1. 0번 고객과 1번 고객의 코사인 유사도 : {cos_a_b:.4f}")
    print(f"2. 자기 자신과의 코사인 유사도 (a, a)   : {cos_self:.4f}")
    print(f"3. 스칼라배와의 코사인 유사도 (a, 3*a)  : {cos_scaled:.4f}")
    print(f"4. 직교 벡터와의 코사인 유사도 (a, c)   : {cos_orthogonal:.4f}\n")

    print("=== [코사인 유사도 값 해석 정리] ===")
    print("1.  1 : 두 벡터의 방향이 완전히 동일함 (구매 패턴/품목 비율이 완전히 일치).")
    print("2.  0 : 두 벡터가 서로 직교함 (공통으로 구매한 품목이 전혀 없음).")
    print("3. -1 : 두 벡터의 방향이 완전히 반대임 (음수 구매량이 없는 일반적 거래 데이터에서는 나타나지 않음).\n")

    # -------------------------------------------------------------
    # [문제 2-2] 유사 고객 상위 5명 찾기
    # -------------------------------------------------------------
    # 1. 전체 고객 간 유사도 행렬 계산 및 shape 확인
    cos_sim_matrix = cosine_similarity(M)
    
    # 2. 고객 ID를 인덱스 및 컬럼으로 하는 DataFrame 생성
    cos_sim_df = pd.DataFrame(
        cos_sim_matrix, 
        index=M_df.index, 
        columns=M_df.index
    )

    # 0번 고객 ID
    target_customer_id = M_df.index[0]
    
    # 3. 0번 고객 기준 유사도 상위 5명 추출 (자기 자신 제외)
    target_cos_series = cos_sim_df.loc[target_customer_id].sort_values(ascending=False)
    top_5_except_self_cos = target_cos_series.iloc[1:6]

    df_top_5_cos = pd.DataFrame({
        "Customer_ID": top_5_except_self_cos.index,
        "코사인 유사도": np.round(top_5_except_self_cos.values, 4),
        "총 구매량": total_quantities.loc[top_5_except_self_cos.index].values
    })

    print("=== [1장-2강] 문제 2-2 출력 결과 ===")
    print(f"1. 코사인 유사도 행렬 shape: {cos_sim_matrix.shape}")
    print(f"\n2. 0번 고객({target_customer_id}) 기준 코사인 유사도 상위 5명 (자기 자신 제외):")
    print(df_top_5_cos.to_string(index=False))

    print("\n3. 문제 1-2 내적 기준 상위 5명과의 비교:")
    print("   - 내적 상위 목록     :", list(df_top_5["Customer_ID"]))
    print("   - 코사인 유사도 상위 목록:", list(df_top_5_cos["Customer_ID"]))

    print("\n=== [지표 선택 근거 (지표 선택 이유)] ===")
    print("코사인 유사도는 벡터의 길이를 L2 노름으로 정규화하여 총 구매량(규모) 차이를 제거하고 "
          "순수한 구매 품목 비율(방향)만 비교하므로, 대량 구매자와 소량 구매자 간의 취향 유사성을 왜곡 없이 파악하는 데 적합합니다.\n")    

    # -------------------------------------------------------------
    # [문제 3-1] 유사 고객 기반 추천 후보 도출하기
    # -------------------------------------------------------------
    # 1. 코사인 유사도가 가장 높은 이웃 고객 1명 선택
    neighbor_id = top_5_except_self_cos.index[0]
    neighbor_sim = top_5_except_self_cos.iloc[0]

    # 2. 기준 고객 및 이웃 고객의 구매량 벡터: 추천 점수로 사용.
    target_vector = M_df.loc[target_customer_id]
    neighbor_vector = M_df.loc[neighbor_id]

    # 3. 기준 고객이 아직 구매하지 않은 상품 (구매량 == 0) 중, 이웃 고객은 구매한 상품 (구매량 > 0) 필터링
    unpurchased_mask = (target_vector == 0)
    neighbor_purchased_mask = (neighbor_vector > 0)

    recommend_candidates = neighbor_vector[unpurchased_mask & neighbor_purchased_mask].sort_values(ascending=False)

    # 4. 상위 10개 (10개 미만이면 전체) 추출
    top_10_candidates = recommend_candidates.head(10)

    df_recommendations = pd.DataFrame({
        "StockCode (상품 코드)": top_10_candidates.index,
        "추천 점수 (이웃 구매량)": top_10_candidates.values
    })

    print("=== [1장-2강] 문제 3-1 출력 결과 ===")
    print(f"1. 선택된 이웃 고객 ID: {neighbor_id} (코사인 유사도: {neighbor_sim:.4f})")
    print(f"2. 유효 추천 후보 상품 개수: {len(recommend_candidates)} 개\n")
    print(f"3. 추천 후보 상품 및 점수 (상위 최대 10개):")
    print(df_recommendations.to_string(index=False))

    print("\n=== [임베딩 기반 유사도 검색과의 원리 연결 설명] ===")
    print("질문(기준 고객) 벡터를 고차원 벡터 공간에 매핑한 뒤 코사인 유사도로 가장 가까운 문서(이웃 고객) 벡터를 검색하고, "
          "해당 문서의 핵심 정보(미구매 상품)를 결과로 추출하는 흐름이 임베딩 검색(k-NN Nearest Neighbor)과 완벽히 동일한 원리입니다.\n")

# 코사인 유사도 직접 구현 함수
def cosine_similarity_single(a, b):
    norm_a = np.linalg.norm(a, 2)
    norm_b = np.linalg.norm(b, 2)
    
    # 0으로 나누기 예방
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return np.dot(a, b) / (norm_a * norm_b)

if __name__ == "__main__":
    main()
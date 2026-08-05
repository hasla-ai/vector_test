import numpy as np
from dataset import get_wine_data

# 공통 난수 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def main():
    # 1. 공통 모듈에서 데이터 로드
    X, y = get_wine_data()

    # 2. 첫 번째 샘플의 측정값을 v1(NumPy 배열)로 추출
    v1 = X.iloc[0].to_numpy()

    # 3. v1의 첫 번째 원소 추출 (스칼라)
    s1 = v1[0]

    # --- 출력 결과 ---
    print("=== [1장-1강-필수1] 문제 1-1 출력 결과 ===")
    print(f"1. 전체 데이터 표 (행렬, X) shape: {X.shape}")
    print(f"2. 첫 번째 샘플 (특성 벡터, v1) shape: {v1.shape}")
    print(f"3. v1의 첫 번째 원소 (스칼라, s1) 값: {s1} (특성명: {X.columns[0]})\n")

    print("=== [세 대상의 역할 구분 설명] ===")
    print("1. 데이터 행렬 (X): 전체 와인 샘플과 이화학 측정값을 통합 관리하는 2차원 공간입니다.")
    print("2. 특성 벡터 (v1): 와인 1개 샘플의 다차원 이화학 측정값 묶음으로, 데이터 공간에서 개별 와인의 위치를 정의합니다.")
    print("3. 스칼라 (s1): 산도(fixed_acidity)와 같은 단일 항목 수치로, 특성 벡터를 구성하는 개별 속성값입니다.")

if __name__ == "__main__":
    main()
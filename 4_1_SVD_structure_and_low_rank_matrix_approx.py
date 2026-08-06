import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_4_1 import get_matrix

# 데이터 로드
M = get_matrix()
m_rows, n_cols = M.shape
print(f"고객-상품 행렬 M Shape: {M.shape}")

# ==========================================
# 필수 1 : 정사각형이 아닌 행렬도 분해하기
# ==========================================
print("\n" + "="*50)
print("[필수 1] 문제 1-1: SVD 적용 및 shape 해석")
print("="*50)

# 1. 축소형 SVD
U, S, Vt = np.linalg.svd(M, full_matrices=False)

# 2. Shape 출력 및 역할 설명
print(f"축소형 SVD -> U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
print("- U: (60, 40) -> 고객 공간의 직교 잠재 특성 벡터 (좌특이벡터)")
print("- S: (40,) -> 각 잠재 축의 중요도를 나타내는 특이값 배열")
print("- Vt: (40, 40) -> 상품 공간의 직교 잠재 특성 벡터 (우특이벡터)")

# 3. 전체형 SVD 비교
Uf, Sf, Vtf = np.linalg.svd(M, full_matrices=True)
print(f"전체형 SVD -> Uf: {Uf.shape}, Sf: {Sf.shape}, Vtf: {Vtf.shape}")

# 4. 정렬 확인 및 상위 5개 특이값
is_sorted = np.all(S[:-1] >= S[1:])
print(f"특이값 내림차순 정렬 여부: {is_sorted}")
print(f"상위 5개 특이값: {S[:5]}")

# 5. S가 1차원 배열인 이유
print("\n[설명 1-1-5] S는 대각 성분 외의 모든 값이 0인 대각행렬이므로 메모리 효율성을 위해 대각 원소만 1차원 배열로 반환합니다.")

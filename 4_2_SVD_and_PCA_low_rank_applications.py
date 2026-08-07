# 4_2_SVD_and_PCA_low_rank_applications.py 파일 최상단에 추가
from dataset_4_2 import M, M_df

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ==========================================
# 필수 1 : PCA와 SVD가 사실 같은 계산이라는 것 확인하기
# ==========================================

# ==========================================
# 문제 1-1 : 중심화 후 SVD의 Vᵀ와 PCA 주성분 비교하기
# ==========================================

# 1. M의 열 평균을 빼서 중심화한 Mc 생성
M_mean = np.mean(M, axis=0)
Mc = M - M_mean
mc_column_means = np.mean(Mc, axis=0)

# 2. np.linalg.svd로 U, S, Vt 구하기
U, S, Vt = np.linalg.svd(Mc, full_matrices=False)

# 3. PCA(n_components=5) 모델 적합
pca = PCA(n_components=5, random_state=42)
pca.fit(M)

# 4. 절댓값 기준 일치 여부 확인
# SVD의 Vt[:5]와 PCA의 components_ 비교 (부호 반전 허용)
components_diff = np.abs(np.abs(Vt[:5]) - np.abs(pca.components_))
is_matched = np.allclose(components_diff, 0, atol=1e-10)

# 출력 결과
print("=== [문제 1-1] 출력 결과 ===")
print(f"1. 중심화 후 열 평균 최대 절대값: {np.max(np.abs(mc_column_means)):.18f}")
print(f"2. Vt[:5]와 pca.components_ 절댓값 일치 여부: {is_matched}")
print("\n3. 두 방식의 첫 번째 주성분 방향 비교 (앞 5개 성분):")
df_comp1 = pd.DataFrame({
    'SVD Vt[0] (앞 5개)': Vt[0][:5],
    'PCA Comp[0] (앞 5개)': pca.components_[0][:5],
    '절댓값 차이': np.abs(np.abs(Vt[0][:5]) - np.abs(pca.components_[0][:5]))
})
print(df_comp1.to_string(index=False))


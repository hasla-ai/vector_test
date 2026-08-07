# 4_2_SVD_and_PCA_low_rank_applications.py 파일 최상단에 추가
from dataset_4_2 import M, M_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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


# ==========================================
# 문제 1-2 : 특이값에서 설명 분산 계산하기
# ==========================================

n_samples = M.shape[0]

# 1. 중심화 후 SVD 수행하여 특이값 S 구하기
Mc = M - np.mean(M, axis=0)
_, S, _ = np.linalg.svd(Mc, full_matrices=False)

# 2. 특이값에서 직접 설명 분산 및 설명분산비 계산
# S^2 / (n_samples - 1)
explained_variance_manual = (S ** 2) / (n_samples - 1)
explained_variance_ratio_manual = explained_variance_manual / np.sum(explained_variance_manual)

# 3. sklearn PCA(n_components=5) 적합
pca = PCA(n_components=5, random_state=42)
pca.fit(M)

# 4. 상위 5개 비교 및 검증
var_matched = np.allclose(explained_variance_manual[:5], pca.explained_variance_, atol=1e-10)
ratio_matched = np.allclose(explained_variance_ratio_manual[:5], pca.explained_variance_ratio_, atol=1e-10)

# 5. 출력 결과 작성
print("=== [문제 1-2] 출력 결과 ===")
print("1. 직접 계산한 설명 분산과 sklearn PCA 비교:")
df_var = pd.DataFrame({
    '직접 계산 (S^2 / (n-1))': explained_variance_manual[:5],
    'sklearn PCA': pca.explained_variance_,
    '차이': np.abs(explained_variance_manual[:5] - pca.explained_variance_)
})
print(df_var.to_string(index=True))

print(f"\n2. 설명분산비 일치 여부: {ratio_matched}")
print(f"3. 상위 5개 주성분의 누적 설명분산비: {np.sum(pca.explained_variance_ratio_[:5]):.4f} ({np.sum(pca.explained_variance_ratio_[:5])*100:.2f}%)")

# ==========================================
# 필수 2: 실무 차원 축소 파이프라인 만들기
# ==========================================
# ==========================================
# 문제 2-1: 4단계 파이프라인 구현하기
# ==========================================

# 1. StandardScaler로 M 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
M_scaled = scaler.fit_transform(M)

# 2. 표준화된 데이터에 SVD 적용
U, S, Vt = np.linalg.svd(M_scaled, full_matrices=False)

# 3. 누적 설명분산비가 80%를 넘는 최소 주성분 개수 k 구하기
var_explained = (S ** 2) / (M_scaled.shape[0] - 1)
var_ratio = var_explained / np.sum(var_explained)
cum_var_ratio = np.cumsum(var_ratio)
k = np.argmax(cum_var_ratio >= 0.80) + 1

# 4. 상위 k개 방향으로 데이터를 투영해 Z 만들기 (Z = M_scaled @ Vt[:k].T)
Z = M_scaled @ Vt[:k].T

# 5. 같은 k로 PCA(n_components=k)를 적용한 결과와 절댓값 비교
pca_k = PCA(n_components=k, random_state=42)
Z_pca = pca_k.fit_transform(M_scaled)
is_proj_matched = np.allclose(np.abs(Z), np.abs(Z_pca), atol=1e-10)

# 출력 결과
print("=== [문제 2-1] 출력 결과 ===")
print(f"1. 선택된 최소 주성분 개수 (k): {k}")
print(f"2. k={k}일 때 누적 설명분산비: {cum_var_ratio[k-1]:.4f} ({cum_var_ratio[k-1]*100:.2f}%)")
print(f"3. 투영 결과 Z shape: {Z.shape}")
print(f"4. 원본 대비 차원 축소율: {(1 - Z.shape[1] / M.shape[1]) * 100:.2f}% ({M.shape[1]}차원 -> {k}차원)")
print(f"5. sklearn PCA 결과와의 절댓값 일치 여부: {is_proj_matched}")

# 6. 투영 결과 2차원 산점도 (PC1, PC2)
plt.figure(figsize=(7, 5))
plt.scatter(Z[:, 0], Z[:, 1], alpha=0.8, color='crimson', edgecolors='k')
plt.title(f'Customer Embeddings in 2D Space (PC1 vs PC2, k={k})', fontsize=12)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# 문제 2-2: 표준화를 생략하면 어떻게 되는지 확인하기
# ==========================================

# 1. 표준화 미적용 (중심화만 진행)
Mc = M - np.mean(M, axis=0)
_, S_raw, Vt_raw = np.linalg.svd(Mc, full_matrices=False)
var_raw = (S_raw ** 2) / (M.shape[0] - 1)
ratio_raw = var_raw / np.sum(var_raw)

# 2. 표준화 적용
scaler = StandardScaler()
M_scaled = scaler.fit_transform(M)
_, S_scaled, Vt_scaled = np.linalg.svd(M_scaled, full_matrices=False)
var_scaled = (S_scaled ** 2) / (M.shape[0] - 1)
ratio_scaled = var_scaled / np.sum(var_scaled)

# 지표 1: PC1 설명분산비
pc1_ratio_raw = ratio_raw[0]
pc1_ratio_scaled = ratio_scaled[0]

# 지표 2: PC1 로딩 집중도 (max |Vt[0]|)
loading_max_raw = np.max(np.abs(Vt_raw[0]))
loading_max_scaled = np.max(np.abs(Vt_scaled[0]))

# 지표 3: 누적 설명분산비 80% 도달에 필요한 k
k_raw = np.argmax(np.cumsum(ratio_raw) >= 0.80) + 1
k_scaled = np.argmax(np.cumsum(ratio_scaled) >= 0.80) + 1

# 3. 표준화 미적용 시 PC1 절댓값 상위 3개 상품 찾기
top3_prod_idx = np.argsort(np.abs(Vt_raw[0]))[::-1][:3]
top3_prod_codes = M_df.columns[top3_prod_idx].tolist()

# 4. 해당 상품들의 원본 구매량 표준편차 순위 확인
std_per_product = np.std(M, axis=0, ddof=1)
std_ranks = np.argsort(np.argsort(-std_per_product)) + 1  # 1등부터 순위 산출

# 출력 결과
print("=== [문제 2-2] 출력 결과 ===")
print(f"1. PC1 설명분산비 비교: [표준화 전] {pc1_ratio_raw:.4f} vs [표준화 후] {pc1_ratio_scaled:.4f}")
print(f"   (설명분산비만 보고는 표준화 필요성을 판별할 수 없음)")
print(f"2. PC1 로딩 집중도 (max|Vt[0]|) 비교: [표준화 전] {loading_max_raw:.4f} vs [표준화 후] {loading_max_scaled:.4f}")
print(f"3. 80% 분산 설명에 필요한 k 비교: [표준화 전] {k_raw}개 vs [표준화 후] {k_scaled}개")

print("\n4. 표준화 미적용 시 PC1 주도 상위 3개 상품 및 원본 표준편차 순위:")
df_top3 = pd.DataFrame({
    'StockCode': top3_prod_codes,
    'PC1 가중치 (Vt[0])': Vt_raw[0][top3_prod_idx],
    '원본 표준편차': std_per_product[top3_prod_idx],
    '전체 60개 중 표준편차 순위': [f"{rank}위" for rank in std_ranks[top3_prod_idx]]
})
print(df_top3.to_string(index=False))

print("\n5. 표준화 생략 시 발생하는 문제점 요약:")
print(" - 표준화를 생략하면 구매량 변동성(표준편차)이 큰 특정 상품이 PC1 가중치를 독점합니다.")
print(" - 이로 인해 주성분이 데이터 전체의 잠재적 패턴이나 유사 구조가 아닌 '스케일이 큰 상품'의 크기만 반영하게 됩니다.")
print(" - 따라서 각 항목이 공평한 비중으로 반영되도록 차원 축소 전 표준화(StandardScaler) 적용이 필수적입니다.")

# ==========================================
# 심화 1. 문제 3-1:
# ==========================================
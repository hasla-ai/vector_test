# 4_3_linear_algebra_mini_project.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# dataset_4_3에서 데이터 로드
from dataset_4_3 import load_bank_data

X_df, y = load_bank_data()

# ====================================================
# 필수 1 : 고객 데이터를 2차원 지도로 만들기
# ====================================================

# ----------------------------------------------------
# 문제 1-1 : 표준화가 필요한지 데이터로 확인하기
# ----------------------------------------------------
print("=== [문제 1-1] 출력 결과 ===")

# 1. 특성별 평균과 표준편차
means = X_df.mean()
stds = X_df.std(ddof=1)
df_summary = pd.DataFrame({'Mean': means, 'Std': stds})
print("\n1. 특성별 평균·표준편차 요약:")
print(df_summary.to_string())

# 2. 표준편차 최대/최소 비율
std_ratio = stds.max() / stds.min()
print(f"\n2. 표준편차 최대/최소 비율: {std_ratio:.2f}")

# 3. StandardScaler로 표준화
scaler = StandardScaler()
Xs = scaler.fit_transform(X_df)
print(f"3. 표준화 후 평균 검증(최대 절댓값): {np.max(np.abs(Xs.mean(axis=0))):.18f}")
print(f"   표준화 후 표준편차 검증(평균): {Xs.std(axis=0).mean():.4f}")

# 4. 표준화 전후 PCA 적용 및 PC1 설명분산비 비교
pca_raw = PCA(n_components=2, random_state=42).fit(X_df)
pca_scaled = PCA(n_components=2, random_state=42).fit(Xs)
print(f"\n4. PC1 설명분산비 비교 -> 표준화 전: {pca_raw.explained_variance_ratio_[0]:.4f} | 표준화 후: {pca_scaled.explained_variance_ratio_[0]:.4f}")

# 5. 이유 서술
print("\n[표준화를 먼저 해야 하는 이유]")
print("표준화를 생략하면 스케일(분산)이 압도적으로 큰 특정 특성이 주성분 축을 독점하여 데이터의 전반적인 잠재 구조를 제대로 반영하지 못하기 때문입니다.")

# ----------------------------------------------------
# 문제 1-2 : PCA로 2차원 지도 그리기
# ----------------------------------------------------
print("\n" + "="*50)
print("=== [문제 1-2] 출력 결과 ===")

# 1. PCA 적용 (Z)
pca_2d = PCA(n_components=2, random_state=42)
Z = pca_2d.fit_transform(Xs)

print(f"1. Z shape: {Z.shape}")
print(f"2. PC1 설명분산비: {pca_2d.explained_variance_ratio_[0]:.4f} | PC2 설명분산비: {pca_2d.explained_variance_ratio_[1]:.4f}")
print(f"   누적 설명분산비: {np.sum(pca_2d.explained_variance_ratio_):.4f} ({np.sum(pca_2d.explained_variance_ratio_)*100:.2f}%)")

# 3. 상위 기여 특성 확인
components_df = pd.DataFrame(pca_2d.components_, columns=X_df.columns, index=['PC1', 'PC2'])
top_pc1 = components_df.loc['PC1'].abs().nlargest(3).index.tolist()
top_pc2 = components_df.loc['PC2'].abs().nlargest(3).index.tolist()
print(f"\n3. PC1 기여 상위 3개 특성: {top_pc1}")
print(f"   PC2 기여 상위 3개 특성: {top_pc2}")

# 4. 시각화 해석 서술
print("\n[시각화 해석]")
print("- 읽을 수 있는 점: 2차원 평면상에서 정기예금 가입 여부(타깃)에 따라 전체 고객 군집이 거시적으로 분리되거나 몰려있는 경향성을 한눈에 파악할 수 있습니다.")
print("- 읽을 수 없는 점: 2차원으로 축소되면서 손실된 나머지 분산 정보로 인해 군집 내부 개별 고객들의 미세한 특성 차이나 정밀한 경계선은 완벽히 파악할 수 없습니다.")

# 5. 산점도 출력
plt.figure(figsize=(7, 5))
scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='coolwarm', alpha=0.5, s=10)
plt.title('2D Customer Map via PCA', fontsize=12)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Target (0: No, 1: Yes)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 문제 2-1 : 원본 공간에서 유사 고객 찾기
# ----------------------------------------------------
print("=== [문제 2-1] 출력 결과 ===")

# 1. 표준화된 데이터 앞 2,000명 샘플링 (Xs_small)
Xs_small = Xs[:2000]

# 2. cosine_similarity로 유사도 행렬 계산 및 시간 측정
start_time = time.time()
sim_orig = cosine_similarity(Xs_small)
calc_time_ms = (time.time() - start_time) * 1000

# 3. 유사도 행렬 shape 및 메모리 사용량(원소 수, nbytes) 확인
matrix_shape = sim_orig.shape
num_elements = sim_orig.size
memory_mb = sim_orig.nbytes / (1024 ** 2)

# 4. 0번 고객과 유사도가 높은 상위 5명 탐색 (자기 자신 인덱스 0 제외)
top5_indices = np.argsort(sim_orig[0])[::-1][1:6]
top5_sim_scores = sim_orig[0][top5_indices]

# 출력 결과
print(f"1. 유사도 행렬 Shape: {matrix_shape}")
print(f"2. 유사도 계산 소요 시간: {calc_time_ms:.2f} ms")
print(f"3. 유사도 행렬 메모리 사용량: {num_elements:,} 개 원소 ({memory_mb:.2f} MB)")

print("\n4. 0번 고객 기준 원본 공간 유사 고객 Top 5:")
df_top5 = pd.DataFrame({
    '고객 인덱스 (Index)': top5_indices,
    '코사인 유사도 (Similarity)': top5_sim_scores
})
print(df_top5.to_string(index=False))

print("\n5. 특성이 수백 개로 늘어날 때의 부담 설명:")
print(" - 특성 수가 수백 개로 늘어나면 유사도 계산 시 특성 차원 축에 대한 연산량이 선형적으로 증가할 뿐만 아니라,")
print("   '차원의 커스(Curse of Dimensionality)' 현상으로 인해 고차원 공간 상의 모든 점들 간 거리/유사도가 평이해져 변별력이 급격히 떨어집니다.")


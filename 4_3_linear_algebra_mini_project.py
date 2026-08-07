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

# ----------------------------------------------------
# 문제 2-2 : 축소 공간에서 유사 고객 찾고 비교하기
# ----------------------------------------------------
print("=== [문제 2-2] 출력 결과 ===")

# 1. 원본 공간 유사도 계산 및 Top 5 추출
start_t = time.time()
sim_orig = cosine_similarity(Xs_small)
t_orig = (time.time() - start_t) * 1000
top5_orig = np.argsort(sim_orig[0])[::-1][1:6]

# 2. 누적 설명분산비 80%를 만족하는 k_search 산출
pca_full = PCA(random_state=42).fit(Xs_small)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
k_search = np.argmax(cum_var >= 0.80) + 1
cum_var_k = cum_var[k_search - 1]

print(f"1. 선택된 k_search 차원 수: {k_search}차원")
print(f"2. k_search 차원의 누적 설명분산비: {cum_var_k:.4f} ({cum_var_k*100:.2f}%)")

# 3. 검색용 k_search차원 및 시각화용 2차원 공간 좌표 생성
Z_search = pca_full.transform(Xs_small)[:, :k_search]
Z_2d = pca_full.transform(Xs_small)[:, :2]

# 4. 각 공간별 유사도 계산 및 시간 측정
start_t = time.time()
sim_search = cosine_similarity(Z_search)
t_search = (time.time() - start_t) * 1000

sim_2d = cosine_similarity(Z_2d)

# 5. 0번 고객 기준 상위 5명 탐색
top5_search = np.argsort(sim_search[0])[::-1][1:6]
top5_2d = np.argsort(sim_2d[0])[::-1][1:6]

# 원본 공간 결과와 겹치는 인원 수
overlap_search = len(set(top5_orig).intersection(set(top5_search)))
overlap_2d = len(set(top5_orig).intersection(set(top5_2d)))

# 전체 유사도 행렬 간 피어슨 상관계수
corr_search = np.corrcoef(sim_orig.ravel(), sim_search.ravel())[0, 1]
corr_2d = np.corrcoef(sim_orig.ravel(), sim_2d.ravel())[0, 1]

# 6. 용량 계산 (MB) 및 절감률
mb_orig = Xs_small.nbytes / (1024 ** 2)
mb_search = Z_search.nbytes / (1024 ** 2)
mb_2d = Z_2d.nbytes / (1024 ** 2)

reduction_search = (1 - mb_search / mb_orig) * 100
reduction_2d = (1 - mb_2d / mb_orig) * 100

# 7. 종합 비교 표 생성
df_compare = pd.DataFrame({
    '공간 구분': [f'원본 ({Xs_small.shape[1]}차원)', f'검색용 ({k_search}차원)', '시각화용 (2차원)'],
    '0번 고객 Top5 Index': [top5_orig.tolist(), top5_search.tolist(), top5_2d.tolist()],
    '원본과 겹치는 수': [5, overlap_search, overlap_2d],
    '유사도 상관계수': [1.0000, corr_search, corr_2d],
    '데이터 용량 (MB)': [mb_orig, mb_search, mb_2d],
    '용량 절감률 (%)': [0.0, reduction_search, reduction_2d]
})

print("\n3. 공간별 유사도 검색 정밀도 및 용량 비교 표:")
print(df_compare.to_string(index=False))

print(f"\n4. 유사도 계산 시간 비교:")
print(f" - 원본 공간: {t_orig:.2f} ms")
print(f" - k_search 차원: {t_search:.2f} ms")
print("   (특성 수 p=7로 작아 계산 연산의 주병목은 샘플 수 N의 제곱 N^2에 위치하므로, 차원 축소에 따른 계산 시간 이득은 미비함)")

print("\n5. 공간 선택 기준 및 상황별 활용 서술:")
print(" - [2차원 시각화 공간 (Z_2d)]: 60% 이상의 정보 손실이 있어 유사 고객 검색용으로는 적합하지 않으나, 마케팅 전략 수립 시 전반적인 고객 군집 분포를 한눈에 파악하는 모니터링 목적으로 사용합니다.")
print(" - [k_search 공간 (Z_search)]: 원본 정보의 80% 이상을 보존하면서 데이터를 효율적으로 압축하므로, 데이터 저장 비용을 줄이면서도 실무 추천 서비스의 정확도를 유지해야 하는 경우 최선의 선택입니다.")

# ----------------------------------------------------
# 문제 3-1 : SVD vs 고유분해 vs sklearn 3가지 방식 구현 및 검증
# ----------------------------------------------------
print("=== [문제 3-1] 출력 결과 ===")

N, p = Xs.shape

# 1) sklearn PCA 방식
pca_sk = PCA(n_components=2, random_state=42)
Z_sk = pca_sk.fit_transform(Xs)
V_sk = pca_sk.components_.T  # Shape: (p, 2)

# 2) Covariance Matrix + Eigendecomposition 방식
cov_matrix = np.cov(Xs, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 내림차순 정렬
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

V_eig = eigenvectors[:, :2]  # 상위 2개 고유벡터
Z_eig = Xs @ V_eig           # 투영 좌표 Z

# 3) SVD (Singular Value Decomposition) 방식 (Xs = U * S * Vt)
U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
V_svd = Vt.T[:, :2]          # 우단수고유벡터 (Right Singular Vectors)
Z_svd = Xs @ V_svd           # 또는 U[:, :2] @ np.diag(S[:2])

# ----------------------------------------------------
# 결과 비교 및 검증 (부호 반전 처리 포함)
# ----------------------------------------------------
# PCA에서 주성분 축의 방향(부호)은 ±1 둘 다 정답이므로 절대값 오차 계산
diff_V_eig = np.abs(np.abs(V_sk) - np.abs(V_eig)).max()
diff_V_svd = np.abs(np.abs(V_sk) - np.abs(V_svd)).max()

diff_Z_eig = np.abs(np.abs(Z_sk) - np.abs(Z_eig)).max()
diff_Z_svd = np.abs(np.abs(Z_sk) - np.abs(Z_svd)).max()

print("1. 주성분 벡터(V) 최대 절대값 차이:")
print(f" - sklearn vs Eigendecomposition : {diff_V_eig:.2e}")
print(f" - sklearn vs SVD               : {diff_V_svd:.2e}")

print("\n2. 투영 좌표(Z) 최대 절대값 차이:")
print(f" - sklearn vs Eigendecomposition : {diff_Z_eig:.2e}")
print(f" - sklearn vs SVD               : {diff_Z_svd:.2e}")

# 3. 주성분 벡터(V_1, V_2) 비교 표 출력
df_vectors = pd.DataFrame({
    'Feature': X_df.columns,
    'V1 (sklearn)': V_sk[:, 0],
    'V1 (Eig)': V_eig[:, 0],
    'V1 (SVD)': V_svd[:, 0],
    'V2 (sklearn)': V_sk[:, 1],
    'V2 (Eig)': V_eig[:, 1],
    'V2 (SVD)': V_svd[:, 1]
})

print("\n3. 주성분 벡터 V 상위 5개 특성 비교 (방향 오차 확인):")
print(df_vectors.head(5).to_string(index=False))

print("\n4. 부호(Sign) 방향성 차이 발생 이유 서술:")
print(" - 고유벡터 v 및 Singular Vector v는 Av = λv 또는 Xs*v = σu 를 만족할 때, 부호를 반대로 뒤집은 (-v) 역시 동일한 식을 만족합니다.")
print(" - 즉, 주성분 축의 '방향성(Line)'이 동일하다면 양/음의 방향 기호는 수학적으로 완전히 동등하므로 결과상의 오류가 아닙니다.")
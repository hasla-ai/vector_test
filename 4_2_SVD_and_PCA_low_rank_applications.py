# 4_2_SVD_and_PCA_low_rank_applications.py 파일 최상단에 추가
from dataset_4_2 import M, M_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
# 심화 1. 저랭크 근사로 추천 후보 만들기
# ==========================================
# ==========================================
# 문제 3-1: TruncatedSVD 임베딩으로 유사 고객과 추천 후보 찾기
# ==========================================

## SVD 분해 결과($U, \Sigma, V^T$)에서 상위 $k=5$개의 성분만 사용해 행렬을 복원($\hat{M}_5$)함으로써, 
## 고객이 아직 구매하지 않았거나 적게 구매한 상품에 대해 잠재적 구매 선호도를 예측하는 코드

# 1. 원본 데이터 M에 대해 SVD 수행 (k=5 저랭크 근사)
# SVD 분해: M ≈ U_k * Sigma_k * Vt_k
U, S, Vt = np.linalg.svd(M, full_matrices=False)

k = 5
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

# 2. k=5 저랭크 근사 행렬 M_hat_5 재구성
M_hat_5 = U_k @ S_k @ Vt_k

# 3. 원본 행렬과 근사 행렬 간 Frobenius Norm 오차 및 오차 비율 계산
f_norm_M = np.linalg.norm(M, 'fro')
f_norm_err = np.linalg.norm(M - M_hat_5, 'fro')
reconstruction_err_ratio = (f_norm_err / f_norm_M) * 100

# 4. 첫 번째 고객(Customer Index 0)의 추천 후보 찾기
cust_0_actual = M[0, :]
cust_0_pred = M_hat_5[0, :]

# 원본에서 구매량이 0이었던 상품들의 마스크 생성
zero_purchase_mask = (cust_0_actual == 0)

# 구매량이 0인 상품 중 복원 값(예측 선호도)이 가장 높은 상위 3개 상품 추출
pred_unpurchased = cust_0_pred.copy()
pred_unpurchased[~zero_purchase_mask] = -np.inf  # 이미 구매한 상품은 제외

top3_rec_idx = np.argsort(pred_unpurchased)[::-1][:3]
top3_rec_codes = M_df.columns[top3_rec_idx].tolist()
top3_rec_scores = cust_0_pred[top3_rec_idx]

# 출력 결과
print("=== [문제 3-1] 출력 결과 ===")
print(f"1. 원본 행렬 M Frobenius Norm: {f_norm_M:.4f}")
print(f"2. k=5 저랭크 근사 복원 오차(Frobenius Norm): {f_norm_err:.4f}")
print(f"3. 상대 재구성 오차 비율: {reconstruction_err_ratio:.2f}%")

print("\n4. 첫 번째 고객(Customer 0) 미구매 상품 중 잠재 선호도 상위 3개 추천 상품:")
df_rec = pd.DataFrame({
    'StockCode': top3_rec_codes,
    '예측 구매 선호도 스코어': top3_rec_scores,
    '실제 구매량': cust_0_actual[top3_rec_idx]
})
print(df_rec.to_string(index=False))

print("\n5. 저랭크 근사 추천의 수학적 의미:")
print(" - 상위 k개 특이값만 남김으로써 잠재 요인(Latent Factors) 기반의 패턴을 학습합니다.")
print(" - 실제 구매량이 0이었더라도 노이즈가 제거된 저차원 공간 상에서 유사한 구매 패턴을 갖는 타 고객의 정보가 반영되어 잠재 선호도가 복원됩니다.")

# ==========================================
# 문제 3-2(자체 응용): 최적 $k$ 탐색 및 Eckart-Young-Mirsky 정리 검증
# ==========================================

# 1. 원본 데이터 M에 대해 SVD 수행
U, S, Vt = np.linalg.svd(M, full_matrices=False)
total_f_norm = np.linalg.norm(M, 'fro')

k_values = [1, 3, 5, 10, 20]
results = []

for k in k_values:
    # 저랭크 근사 행렬 구성
    M_hat_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    # 1) 실제 Frobenius norm 오차
    actual_err = np.linalg.norm(M - M_hat_k, 'fro')
    
    # 2) Eckart-Young 정리에 따른 이론적 오차: sqrt(sum(S[k:]^2))
    theoretical_err = np.sqrt(np.sum(S[k:] ** 2))
    
    # 오차 일치 여부 확인
    is_matched = np.isclose(actual_err, theoretical_err, atol=1e-10)
    
    # 누적 설명분산비 (S^2 기반)
    cum_var_ratio = np.sum(S[:k] ** 2) / np.sum(S ** 2)
    
    results.append({
        'k': k,
        '실제 복원 오차 (||M - M_hat||_F)': actual_err,
        '이론적 오차 (sqrt(sum(S_i^2)))': theoretical_err,
        '이론 일치 여부': is_matched,
        '상대 복원 오차율 (%)': (actual_err / total_f_norm) * 100,
        '누적 설명 분산 비율 (%)': cum_var_ratio * 100
    })

df_results = pd.DataFrame(results)

# 출력 결과
print("=== [문제 3-2] 출력 결과 ===")
print("1. k 변화에 따른 Eckart-Young-Mirsky 정리 검증 결과:")
print(df_results[['k', '실제 복원 오차 (||M - M_hat||_F)', '이론적 오차 (sqrt(sum(S_i^2)))', '이론 일치 여부', '누적 설명 분산 비율 (%)']].to_string(index=False))

# 2. k에 따른 복원 오차 및 설명분산비 그래프 시각화
fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:red'
ax1.set_xlabel('Rank k', fontsize=11)
ax1.set_ylabel('Reconstruction Error (Frobenius Norm)', color=color, fontsize=11)
ax1.plot(df_results['k'], df_results['실제 복원 오차 (||M - M_hat||_F)'], marker='o', color=color, linewidth=2, label='Reconstruction Error')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Cumulative Variance Explained (%)', color=color, fontsize=11)
ax2.plot(df_results['k'], df_results['누적 설명 분산 비율 (%)'], marker='s', color=color, linewidth=2, linestyle='--', label='Cum Variance (%)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Low-Rank Approximation Performance vs. Rank k', fontsize=13)
fig.tight_layout()
plt.show()

print("\n2. 결론 및 정리:")
print(" - Eckart-Young-Mirsky 정리에 의해 k개 성분으로 절단한 저랭크 근사의 최적 오차는 나머지 특이값들의 제곱합 루트와 완벽히 일치합니다.")
print(" - k가 증가함에 따라 복원 오차는 단조 감소하며, 정보 손실과 모델 복잡도 사이의 트레이드오프를 고려해 적절한 k를 선정을 진행합니다.")

# ==========================================
# 문제 3-3(자체 응용): 성능 평가 코드 (Train/Test Split & RMSE/MAE)
# ==========================================

print("=== [문제 3-3] 성능 평가 코드 (Train/Test Split & RMSE/MAE) 출력 결과 ===")

def evaluate_low_rank_recommendation(data, k_components=[2, 5, 8, 12, 16, 20], test_ratio=0.2, seed=42):
    np.random.seed(seed)
    train_data = data.copy()
    
    # 구매가 일어난 Cell(Non-zero)의 위치 탐색
    non_zero_indices = np.argwhere(data > 0)
    n_test = int(len(non_zero_indices) * test_ratio)
    
    test_mask_idx = np.random.choice(len(non_zero_indices), size=n_test, replace=False)
    test_indices = non_zero_indices[test_mask_idx]
    
    # Train 세트를 만들기 위해 마스킹 (0으로 변환)
    for r, c in test_indices:
        train_data[r, c] = 0
        
    eval_results = []
    
    for k_val in k_components:
        # TruncatedSVD 적합
        svd = TruncatedSVD(n_components=k_val, random_state=seed)
        train_reconstructed = svd.fit_transform(train_data) @ svd.components_
        
        # Train & Test 실제값 및 예측값 수집
        train_true = train_data[train_data > 0]
        train_pred = train_reconstructed[train_data > 0]
        
        test_true = data[test_indices[:, 0], test_indices[:, 1]]
        test_pred = train_reconstructed[test_indices[:, 0], test_indices[:, 1]]
        
        # 오차 계산 (RMSE & MAE)
        train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
        test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
        test_mae = mean_absolute_error(test_true, test_pred)
        
        eval_results.append({
            'k': k_val,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae
        })
        
    return pd.DataFrame(eval_results)

eval_df = evaluate_low_rank_recommendation(M, k_components=[2, 5, 8, 12, 16, 20])
print(eval_df.to_string(index=False))

# 최적 k 선택 및 가이드 안내
best_k = eval_df.loc[eval_df['Test RMSE'].idxmin(), 'k']
print(f"\n★ 최소 Test RMSE를 기록한 최적의 k (Optimal k): {int(best_k)}")
print("="*50)

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(8, 5))

# 1. 왼쪽 Y축: Train RMSE (초록색)
color_train = 'tab:green'
ax1.set_xlabel('Latent Factor k', fontsize=11, fontweight='bold')
ax1.set_ylabel('Train RMSE', color=color_train, fontsize=11, fontweight='bold')
line1 = ax1.plot(eval_df['k'], eval_df['Train RMSE'], marker='o', color=color_train, 
                 linewidth=2, label='Train RMSE (Underfitting check)')
ax1.tick_params(axis='y', labelcolor=color_train)
ax1.grid(True, linestyle='--', alpha=0.5)

# 2. 오른쪽 Y축: Test RMSE (빨간색)
ax2 = ax1.twinx()
color_test = 'tab:red'
ax2.set_ylabel('Test RMSE', color=color_test, fontsize=11, fontweight='bold')
line2 = ax2.plot(eval_df['k'], eval_df['Test RMSE'], marker='s', color=color_test, 
                 linewidth=2, linestyle='--', label='Test RMSE (Overfitting check)')
ax2.tick_params(axis='y', labelcolor=color_test)

# 3. 최적의 k=8 지점 강조 표시 (Star Marker & Annotation)
best_k_val = int(best_k)
best_test_rmse = eval_df.loc[eval_df['k'] == best_k_val, 'Test RMSE'].values[0]

ax2.plot(best_k_val, best_test_rmse, marker='*', markersize=15, color='gold', 
         markeredgecolor='black', label=f'Optimal k={best_k_val}')
ax2.annotate(f'Optimal k={best_k_val}\n(Min Test RMSE: {best_test_rmse:.1f})',
             xy=(best_k_val, best_test_rmse),
             xytext=(best_k_val + 1.5, best_test_rmse + 5),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

# 4. 범례 및 제목 통합
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True)

plt.title('Train vs Test RMSE by Latent Factor k (Bias-Variance Tradeoff)', fontsize=13, y=1.18, fontweight='bold')
fig.tight_layout()
plt.show()
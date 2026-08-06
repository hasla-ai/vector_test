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

# ==========================================
# 문제 1-1 : SVD를 적용하고 각 행렬의 shape 해석하기
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

# ==========================================
# 문제 1-2 : 직교성 확인과 원본 복원하기
# ==========================================

print("\n" + "="*50)
print("[필수 1] 문제 1-2: 직교성 확인과 원본 복원")
print("="*50)

# 1. 직교성 검증
UtU = U.T @ U
VtVtT = Vt @ Vt.T
print(f"U^T @ U 가 단위행렬인가? {np.allclose(UtU, np.eye(40))}")
print(f"Vt @ Vt^T 가 단위행렬인가? {np.allclose(VtVtT, np.eye(40))}")

# 2. 원본 복원 (축소형 사용)
Sigma = np.diag(S)
M_reconstructed = U @ Sigma @ Vt

# 3, 4. 복원 오차 및 상대 오차
rec_error = np.linalg.norm(M - M_reconstructed, 'fro')
orig_norm = np.linalg.norm(M, 'fro')
rel_error = rec_error / orig_norm

print(f"복원 오차(Frobenius Norm): {rec_error:.12e}")
print(f"상대 오차: {rel_error:.12e}")

# 5. 완전 복원 설명
print("\n[설명 1-2-5] 정규직교 기저 U, V와 모든 특이값을 사용한 UΣVᵀ 계산은 수학적으로 원본 행렬과 완전 동일하므로 남은 오차는 부동소수점 연산 오차뿐입니다.")

# ==========================================
# 필수 2 : 특이값은 어디에서 오는가
# ==========================================
# ==========================================
# 문제 2-1 : 특이값과 AᵀA 고유값의 관계 확인하기
# ==========================================

print("\n" + "="*50)
print("[필수 2] 문제 2-1: 특이값과 M^T M 고유값의 관계")
print("="*50)

# 1. M^T M 계산 및 대칭성 확인
MTM = M.T @ M
is_symmetric = np.allclose(MTM, MTM.T)
print(f"M^T M 이 대칭행렬인가? {is_symmetric}")

# 2, 3, 4. 고유값 계산 및 특이값 비교
eigvals = np.linalg.eigvalsh(MTM)
eigvals_sorted = np.sort(eigvals)[::-1]
sqrt_eigvals = np.sqrt(np.clip(eigvals_sorted, 0, None))

df_comp = pd.DataFrame({
    'Eigenvalue Sqrt': sqrt_eigvals[:5],
    'Singular Value (S)': S[:5],
    'Diff': np.abs(sqrt_eigvals[:5] - S[:5])
})
print("\n[상위 5개 비교 표]")
print(df_comp.to_string(index=False))
print(f"전체 최대 차이: {np.max(np.abs(sqrt_eigvals - S)):.12e}")

# 5. clip 적용 이유 설명
print("\n[설명 2-1-5] 수치 계산 시 0에 가까운 고유값이 오차로 인해 -1e-16 등 미세한 음수가 될 수 있으며, 이에 np.sqrt를 적용할 때 발생하는 nan 오류를 방지하기 위해 clip을 사용합니다.")

# 6, 7. 랭크 결손 행렬 R 실험
rng = np.random.default_rng(42)
C = rng.normal(size=(60, 5))
P = rng.normal(size=(5, 40))
R = C @ P

RTR = R.T @ R
eig_RTR = np.sort(np.linalg.eigvalsh(RTR))[::-1]

neg_eigs = eig_RTR[eig_RTR < 0]
print(f"\n랭크 결손 행렬 R의 rank: {np.linalg.matrix_rank(R)}")
print(f"R^T R 의 음수 고유값 개수: {len(neg_eigs)}개 (예: {neg_eigs[0] if len(neg_eigs)>0 else 0:.2e})")

# clip 없이 vs clip 적용 np.sqrt
with np.errstate(invalid='ignore'):
    sqrt_no_clip = np.sqrt(eig_RTR)
nan_count_no_clip = np.isnan(sqrt_no_clip).sum()
sqrt_with_clip = np.sqrt(np.clip(eig_RTR, 0, None))
nan_count_with_clip = np.isnan(sqrt_with_clip).sum()

print(f"clip 미적용 시 nan 개수: {nan_count_no_clip}")
print(f"clip 적용 시 nan 개수: {nan_count_with_clip}")

# ==========================================
# 문제 2-2 : 고유분해와 SVD의 적용 범위 비교하기
# =========================================

# 심화 1 : 상위 몇 개만 남겨도 원본을 설명할 수 있을까
# ==========================================
# 문제 3-1 : 저랭크 근사의 오차와 정보 보존량 계산하기
# ==========================================
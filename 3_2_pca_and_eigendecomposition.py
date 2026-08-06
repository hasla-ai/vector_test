import os
import numpy as np
import pandas as pd
import platform
import logging
import matplotlib.pyplot as plt
from numpy.linalg import inv, det, matrix_rank, cond
from sklearn.decomposition import PCA

from dataset_3_2 import X_df, Xs, y_raw, DATA_SOURCE

# Matplotlib의 font_manager 경고 로그 숨기기 (경고 메시지 완전히 제거)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# 현재 OS에 맞는 폰트 자동 지정
os_name = platform.system()
if os_name == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'    # 맑은 고딕
elif os_name == 'Darwin':                            # macOS
    plt.rcParams['font.family'] = 'AppleGothic'     # 애플고딕
else:                                                # Linux / Colab
    plt.rcParams['font.family'] = 'NanumGothic'     # 나눔고딕

plt.rcParams['axes.unicode_minus'] = False          # 마이너스 기호 깨짐 방지

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams['axes.unicode_minus'] = False
# =========================================================
# [문제 1-1] A = PDP⁻¹ 구성하고 복원하기
# =========================================================
def run_problem_1_1():
    print("=== [문제 1-1] A = PDP⁻¹ 구성하고 복원하기 ===")
    A = np.array([[4.0, 2.0], [1.0, 3.0]])

    # 1. 고유분해
    evals, evecs = np.linalg.eig(A)
    P = evecs
    D = np.diag(evals)

    print("1. P 및 D 행렬")
    print("• P (고유벡터 행렬):\n", P)
    print("• D (고유값 대각행렬):\n", D)

    # 2. 복원 및 검증
    A_reconstructed = P @ D @ inv(P)
    is_same = np.allclose(A, A_reconstructed)

    print("\n2. PDP⁻¹ 복원 검증")
    print("• PDP⁻¹ 계산 결과:\n", A_reconstructed)
    print("• 원본 A와 일치 여부:", is_same)

    # 3. det(P) 및 가역성 확인
    det_P = det(P)
    print(f"\n3. det(P) 값: {det_P:.6f} (≠ 0 이므로 역행렬 P⁻¹ 존재 함)")

    # 4. 설명
    print("\n4. 대각화의 의미")
    print(
        "대각화는 행렬 A를 '좌표축을 바꾸고(P⁻¹) -> 축별로 배율만 곱하고(D) -> 원래 좌표계로 되돌리는(P)' "
        "세 단계로 분해하는 것입니다.\n"
    )

# =========================================================
# [문제 1-2] 대각화로 거듭제곱 계산하고, 불가능한 경우 확인하기
# =========================================================
def run_problem_1_2():
    print("=== [문제 1-2] 대각화로 거듭제곱 계산하고, 불가능한 경우 확인하기 ===")
    A = np.array([[4.0, 2.0], [1.0, 3.0]])
    evals_A, evecs_A = np.linalg.eig(A)
    P_A = evecs_A
    D_A = np.diag(evals_A)

    # 1. A² 및 A¹⁰ 검증
    A2_direct = A @ A
    A2_diag = P_A @ (D_A ** 2) @ inv(P_A)
    is_A2_same = np.allclose(A2_direct, A2_diag)

    A10_direct = np.linalg.matrix_power(A, 10)
    A10_diag = P_A @ (D_A ** 10) @ inv(P_A)
    is_A10_same = np.allclose(A10_direct, A10_diag)

    print("1. 거듭제곱 검증 결과")
    print("• A²  일치 여부:", is_A2_same)
    print("• A¹⁰ 일치 여부:", is_A10_same)

    # 2. 결함 행렬 B = [[1, 1], [0, 1]] 고유분해 및 사전 판정
    B = np.array([[1.0, 1.0], [0.0, 1.0]])
    evals_B, evecs_B = np.linalg.eig(B)
    P_B = evecs_B
    D_B = np.diag(evals_B)

    rank_P = matrix_rank(P_B)
    cond_P = cond(P_B)

    print("\n2. 결함 행렬 B의 고유분해 및 사전 판정")
    print("• B의 고유값:", evals_B)
    print("• B의 고유벡터 행렬 P_B:\n", P_B)
    print(f"• rank(P_B) : {rank_P} (2차원 행렬에서 rank 2 미만이므로 결함 행렬)")
    print(f"• cond(P_B) : {cond_P:.2e} (매우 큰 조건수로 인해 수치적 불안정)")

    # 3. 복원 시도 및 수치적 조용한 오답 확인
    B_reconstructed = P_B @ D_B @ inv(P_B)
    is_B_same = np.allclose(B, B_reconstructed)

    print("\n3. B의 PDP⁻¹ 복원 시도 결과")
    print("• P_B @ D_B @ inv(P_B) 복원 값:\n", B_reconstructed)
    print("• 원본 B와 일치 여부:", is_B_same)
    print("  (NumPy 예외 없이 계산되지만 원본 B가 아닌 단위행렬로 '틀린 값' 출력)")

    # 4. 대칭행렬 설명
    print("\n4. 대칭행렬의 대각화 보장")
    print(
        "대칭행렬은 항상 n개의 직교하는 고유벡터를 가지므로 P가 항상 가역(P⁻¹ = Pᵀ)이 되어 "
        "대각화가 문제없이 보장됩니다.\n"
    )

# =========================================================
# [문제 2-1] PCA를 NumPy로 직접 구현하기
# =========================================================
def run_problem_2_1():
    print("=== [문제 2-1] PCA를 NumPy로 직접 구현하기 ===")

    # 1. 공분산행렬 및 고유분해 (내림차순 정렬)
    cov_matrix = np.cov(Xs, rowvar=False)
    evals, evecs = np.linalg.eigh(cov_matrix)

    sort_idx = np.argsort(evals)[::-1]
    evals_sorted = evals[sort_idx]
    evecs_sorted = evecs[:, sort_idx]

    # 2. 상위 2개 고유벡터로 데이터 투영
    top2_evecs = evecs_sorted[:, :2]
    Z_manual = Xs @ top2_evecs

    # 3. 설명분산비 계산
    exp_var_ratio = evals_sorted / np.sum(evals_sorted)
    top2_var_ratio = exp_var_ratio[:2]
    cum_var_ratio = np.sum(top2_var_ratio)

    # 4. 투영 축 간 상관계수
    corr_z = np.corrcoef(Z_manual[:, 0], Z_manual[:, 1])[0, 1]

    print("1. 수동 구현 결과")
    print("• 투영 데이터 Z_manual shape:", Z_manual.shape)
    print(f"• PC1, PC2 설명분산비: {top2_var_ratio[0]:.4f}, {top2_var_ratio[1]:.4f}")
    print(f"• 상위 2개 누적 설명분산비: {cum_var_ratio:.4f}")
    print(f"• PC1과 PC2 간 상관계수: {corr_z:.6f} (≈ 0, 서로 직교함을 입증)\n")

    return Z_manual, evecs_sorted, exp_var_ratio

# =========================================================
# [문제 2-2] sklearn PCA와 결과 비교하기 (독립 실행 가능하도록 수정)
# =========================================================
def run_problem_2_2(Z_manual=None, evecs_manual=None):
    print("=== [문제 2-2] sklearn PCA와 결과 비교하기 ===")

    # 전달받은 값이 없을 경우 내부에서 직접 계산 (NameError 방지)
    if Z_manual is None or evecs_manual is None:
        cov_matrix = np.cov(Xs, rowvar=False)
        evals, evecs = np.linalg.eigh(cov_matrix)
        sort_idx = np.argsort(evals)[::-1]
        evecs_manual = evecs[:, sort_idx]
        Z_manual = Xs @ evecs_manual[:, :2]

    # 1. sklearn PCA 적용 (n_components=2)
    pca_2 = PCA(n_components=2, random_state=RANDOM_STATE)
    Z_sklearn = pca_2.fit_transform(Xs)

    # 2. 설명분산비 비교
    print("1. 설명분산비 비교")
    print("• sklearn 설명분산비:", np.round(pca_2.explained_variance_ratio_, 4))

    # 3. 절댓값 기준 주성분 방향 및 투영 결과 비교
    vecs_match = np.allclose(np.abs(pca_2.components_), np.abs(evecs_manual[:, :2].T))
    z_match = np.allclose(np.abs(Z_sklearn), np.abs(Z_manual))

    print("\n2. 절댓값 기준 일치 여부")
    print("• 주성분 방향(고유벡터) 절댓값 일치 여부:", vecs_match)
    print("• 투영 결과(Z) 절댓값 일치 여부:", z_match)

    # 4. 전체 주성분 부호 비교 표 작성
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(Xs)

    sign_checks = []
    for i in range(Xs.shape[1]):
        sk_v = pca_full.components_[i]
        mn_v = evecs_manual[:, i]

        is_same_sign = np.allclose(sk_v, mn_v)
        is_opp_sign = np.allclose(sk_v, -mn_v)

        if is_same_sign:
            status = "동일 (+)"
        elif is_opp_sign:
            status = "반전 (-)"
        else:
            status = "불일치"
        sign_checks.append({"주성분": f"PC{i+1}", "부호 관계": status})

    df_sign = pd.DataFrame(sign_checks)
    print("\n3. 전체 주성분별 부호 비교 표")
    print(df_sign.to_string(index=False))

    # 5. 2D 산점도 시각화
    save_path = os.path.join('./images', 'chapter_3_2_problem_2_2_pca_and_eigendecomposition.png')
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(Z_sklearn[:, 0], Z_sklearn[:, 1], c=y_raw, cmap='viridis', alpha=0.8, edgecolors='k', s=40)
    plt.colorbar(scatter, label='Target / Quality')
    plt.title('PCA 2D Projection (Wine Data)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print("\n4. 시각화")
    print("• 2D 산점도 저장 완료: chapter_3_2_problem_2_2_pca_and_eigendecomposition.png\n")

if __name__ == "__main__":
    run_problem_2_2()
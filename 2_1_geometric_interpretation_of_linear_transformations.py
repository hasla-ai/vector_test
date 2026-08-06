import numpy as np
from dataset_2_1 import X2, apply_T, plot_pair

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    # 필수 1. 진단 데이터 분포를 늘리고, 뒤집고, 눌러 보기
    # -------------------------------------------------------------
    # [문제 1-1] 스케일링 행렬 적용하기
    # -------------------------------------------------------------
    print("=== [2장-1강] 문제 1-1 출력 결과 ===")

    # 1. x축 2배, y축 0.8배 스케일링 행렬 S 정의
    S = np.array([[2.0, 0.0], [0.0, 0.8]])

    print("[행렬 X2]")
    print(np.round(X2, 4))

    # 2. apply_T를 활용한 선형 변환 계산
    X2_scaled = apply_T(S, X2)

    print("[행렬 X2_scaled]")
    print(np.round(X2_scaled,4))

    # 3. 변환 전후 각 축 표준편차 비교
    std_before = np.std(X2, axis=0)
    std_after = np.std(X2_scaled, axis=0)

    print("=== 축별 표준편차 변화 ===")
    print(
        f"변환 전 표준편차 (x축, y축) : [{std_before[0]:.4f}, {std_before[1]:.4f}]"
    )
    print(
        f"변환 후 표준편차 (x축, y축) : [{std_after[0]:.4f}, {std_after[1]:.4f}]"
    )
    print(
        f"표준편차 변화 비율         : [{std_after[0]/std_before[0]:.2f}배, {std_after[1]/std_before[1]:.2f}배]\n"
    )

    # 4. 대각 원소의 의미 설명
    print("=== 대각 원소의 의미 설명 ===")
    print(
        "스케일링 행렬의 대각 원소 S[0, 0]=2.0은 x축 방향의 확대/축소 비율을, S[1, 1]=0.8은 y축 방향의 확대/축소 비율을 독립적으로 결정합니다.\n"
    )

    # 5. 시각화 출력 (산점도)
    print("[안내] 변환 전후 비교 산점도 창을 띄웁니다.")
    plot_pair(
        X2,
        X2_scaled,
        label_after="scaled (x*2, y*0.8)",
        title="Scaling Transformation (S = diag([2, 0.8]))",
    )

    # -------------------------------------------------------------
    # [문제 1-2] 반사와 투영 행렬 적용하기
    # -------------------------------------------------------------

    print("=== [2장-1강] 문제 1-2 출력 결과 ===")

    # 1. 반사 행렬 F (x축 기준 반사: y값의 부호만 반전)
    F = np.array([[1.0, 0.0], [0.0, -1.0]])
    X2_reflected = apply_T(F, X2)

    # 반사 전후 L2 노름(원점과의 거리) 보존 확인
    norm_before = np.linalg.norm(X2, axis=1)
    norm_reflected = np.linalg.norm(X2_reflected, axis=1)
    is_norm_preserved = np.allclose(norm_before, norm_reflected)

    print("--- [1] 반사 행렬 (F) 검증 ---")
    print(f"반사 전후 원점과의 거리(L2 노름) 유지 여부 : {is_norm_preserved}\n")

    # 2. 투영 행렬 P (x축 위로 누름: y값을 0으로 만듦)
    P = np.array([[1.0, 0.0], [0.0, 0.0]])
    X2_projected = apply_T(P, X2)

    # 투영 후 두 번째 축(y축) 값이 모두 0인지 확인
    is_y_all_zero = np.allclose(X2_projected[:, 1], 0.0)

    print("--- [2] 투영 행렬 (P) 검증 ---")
    print(f"투영 결과의 y축 값이 모두 0인지 여부       : {is_y_all_zero}\n")

    # 3. 되돌림 가능 여부 및 연산 합성 확인 (F @ F == I, P @ P == I)
    F_sq = F @ F
    P_sq = P @ P
    is_F_inv = np.allclose(F_sq, np.eye(2))
    is_P_inv = np.allclose(P_sq, np.eye(2))

    print("--- [3] 합성 연산 및 되돌림 가능 여부 (Self-Inverse) ---")
    print(f"F @ F == I 여부 (두 번 반사시 원복)            : {is_F_inv}")
    print(
        f"P @ P == I 여부 (두 번 투영시 원복)            : {is_P_inv}  (실제 P @ P = P)"
    )

    # 4. 서로 다른 두 점 [1, 2]와 [1, -5] 투영 비교
    pt1 = np.array([1.0, 2.0])
    pt2 = np.array([1.0, -5.0])

    pt1_proj = apply_T(P, pt1)
    pt2_proj = apply_T(P, pt2)

    print("\n--- [4] 서로 다른 두 점 투영 결과 비교 ---")
    print(f"점 [1, 2]  투영 결과 : {pt1_proj}")
    print(f"점 [1, -5] 투영 결과 : {pt2_proj}")
    print(f"두 투영 결과의 동일 여부 : {np.allclose(pt1_proj, pt2_proj)}\n")

    # 5. 반사와 투영의 차이 (정보 보존 여부) 서술
    print("=== 반사와 투영의 차이 (정보 보존 여부) ===")
    print(
        "반사(Reflection)는 공간의 길이와 각도를 보존하며 자기 자신을 역행렬로 가져(F @ F = I) 원본 복원이 가능한 반면, "
        "투영(Projection)은 특정 차원(y축)의 정보를 완전히 0으로 눌러 파괴하므로 서로 다른 두 점이 같은 점으로 겹치고 역행렬이 존재하지 않아 원본 복원이 불가능합니다."
    )

    # 시각화 출력 (산점도)
    print("\n[안내] 반사 변환 전후 비교 산점도 창을 띄웁니다.")
    plot_pair(
        X2,
        X2_reflected,
        label_after="reflected (F)",
        title="Reflection Transformation (F)",
    )

    print("[안내] 투영 변환 전후 비교 산점도 창을 띄웁니다.")
    plot_pair(
        X2,
        X2_projected,
        label_after="projected (P)",
        title="Projection Transformation (P)",
    )

    # 필수 2: 회전과 합성 변환, 그리고 순서 문제
    # -------------------------------------------------------------
    # [문제 2-1] 회전 행렬 만들고 적용하기
    # -------------------------------------------------------------
    print("=== [2장-1강] 문제 2-1 출력 결과 ===")

    # 1. 45도 회전 행렬 R 정의 (deg2rad 적용)
    theta = np.deg2rad(45)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # 2. 검증용 벡터 (1, 0) 회전 적용
    v = np.array([1.0, 0.0])
    v_rot = apply_T(R, v)

    print("--- [1] 검증용 벡터 (1, 0) 회전 결과 ---")
    print(f"회전 전 벡터 v      : {v}")
    print(
        f"45도 회전 후 벡터   : [{v_rot[0]:.4f}, {v_rot[1]:.4f}] (목표값: [0.7071, 0.7071])\n"
    )

    # 3. 데이터 X2 전체에 회전 적용
    X2_rotated = apply_T(R, X2)

    # 4. 회전 전후 원점까지의 거리(L2 노름) 보존 확인
    norm_before = np.linalg.norm(X2, axis=1)
    norm_after = np.linalg.norm(X2_rotated, axis=1)
    is_norm_preserved = np.allclose(norm_before, norm_after)

    print("--- [2] 거리(L2 노름) 보존 여부 검증 ---")
    print(f"회전 전후 원점과의 거리 유지 여부 : {is_norm_preserved}\n")

    # 5. RᵀR 계산 및 직교행렬(Orthogonal Matrix) 확인
    R_T_R = R.T @ R
    is_identity = np.allclose(R_T_R, np.eye(2))

    print("--- [3] RᵀR 연산 및 단위행렬 검증 ---")
    print(f"RᵀR 계산 결과 :\n{np.round(R_T_R, 4)}")
    print(f"RᵀR == I (단위행렬) 성립 여부 : {is_identity}\n")

    # 6. RᵀR = I 의 의미 설명
    print("=== RᵀR = I 의 의미 설명 ===")
    print(
        "회전 행렬 R은 전치행렬이 곧 역행렬(Rᵀ = R⁻¹) 역할을 하는 직교행렬(Orthogonal Matrix)이며, 이는 공간 변환 시 벡터 간의 각도와 원점으로부터의 길이를 완벽히 보존하는 강체 변환(Rigid Transformation)임을 의미합니다."
    )

    # 7. 시각화 출력
    print("\n[안내] 45도 회전 변환 전후 비교 산점도 창을 띄웁니다.")
    plot_pair(
        X2,
        X2_rotated,
        label_after="rotated (45 deg)",
        title="Rotation Transformation (R_45)",
    )
    # -------------------------------------------------------------
    # [문제 2-2] 합성 변환의 순서 비교하기
    # -------------------------------------------------------------

    print("=== [2장-1강] 문제 2-2 출력 결과 ===")

    # 1. 변환 행렬 S (스케일링) 및 R (45도 회전) 정의
    S = np.array([[2.0, 0.0], [0.0, 0.8]])
    theta = np.deg2rad(45)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # 2. 합성 행렬 M1(스케일링 후 회전) 및 M2(회전 후 스케일링) 생성
    # 벡터 v에 S 적용 후 R 적용: R @ (S @ v) = (R @ S) @ v
    M1 = R @ S  # 스케일링 후 회전
    M2 = S @ R  # 회전 후 스케일링

    print("--- [1] 합성 행렬 비교 ---")
    print(f"M1 (스케일링 후 회전, R @ S) :\n{np.round(M1, 4)}")
    print(f"M2 (회전 후 스케일링, S @ R) :\n{np.round(M2, 4)}")

    is_matrices_equal = np.allclose(M1, M2)
    print(f"M1 == M2 행렬 동일 여부 : {is_matrices_equal}\n")

    # 3. 데이터 X2에 두 합성 변환 각각 적용
    X2_M1 = apply_T(M1, X2)
    X2_M2 = apply_T(M2, X2)

    is_transformed_equal = np.allclose(X2_M1, X2_M2)
    print("--- [2] 변환 결과 비교 ---")
    print(f"X2_M1 == X2_M2 변환 결과 동일 여부 : {is_transformed_equal}\n")

    # 4. 합성 순서가 BA가 되는 이유 설명
    print("=== 합성 순서가 BA가 되는 이유 ===")
    print(
        "벡터 v에 변환 A를 먼저 적용한 결과 (Av)에 변환 B를 적용하면 B(Av) = (BA)v가 되므로, 입력 벡터에 가까운 오른쪽 연산자부터 순차적으로 작용하여 전체 합성 행렬은 BA가 됩니다."
    )

    # 5. 시각화 출력 (두 합성 결과를 한 화면에서 비교)
    print("\n[안내] 두 합성 변환(M1 vs M2) 결과 비교 산점도 창을 띄웁니다.")
    plot_pair(
        X2_M1,
        X2_M2,
        label_after="M2: Rotate -> Scale (S @ R)",
        title="Composite Transformation Order: M1 (R@S) vs M2 (S@R)",
    )

    # 심화 1: 이 변환은 정말 선형 변환일까?
    # -------------------------------------------------------------
    # [문제 3-1] 가산성·동차성으로 선형 변환 판별하기
    # -------------------------------------------------------------

    print("=== [2장-1강] 문제 3-1 출력 결과 ===")

    # 1. 합성 변환 행렬 M1 (R @ S) 및 bias 정의
    S = np.array([[2.0, 0.0], [0.0, 0.8]])
    theta = np.deg2rad(45)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    M1 = R @ S
    bias = np.array([3.0, -2.0])

    # 2. 두 변환 함수 정의
    def T_linear(v):
        return apply_T(M1, v)

    def T_affine(v):
        return apply_T(M1, v) + bias

    # 3. 검증용 무작위 벡터 u, v 및 스칼라 c, 원점 0 설정
    u = np.array([1.5, -0.5])
    v = np.array([-2.0, 3.0])
    c = 2.5
    zero_vec = np.array([0.0, 0.0])

    # 4. T_linear 검증
    lin_add = np.allclose(T_linear(u + v), T_linear(u) + T_linear(v))
    lin_hom = np.allclose(T_linear(c * u), c * T_linear(u))
    lin_zero = np.allclose(T_linear(zero_vec), zero_vec)

    # 5. T_affine 검증
    aff_add = np.allclose(T_affine(u + v), T_affine(u) + T_affine(v))
    aff_hom = np.allclose(T_affine(c * u), c * T_affine(u))
    aff_zero = np.allclose(T_affine(zero_vec), zero_vec)

    # 6. 검증 결과 출력 표 정리
    print("--- [1] 가산성 / 동차성 / 원점 보존 검증 표 ---")
    print(
        f"{'검증 항목':<18} | {'T_linear (선형 변환)':<20} | {'T_affine (아핀 변환)':<20}"
    )
    print("-" * 65)
    print(f"{'1. 가산성 T(u+v)=T(u)+T(v)':<18} | {str(lin_add):<20} | {str(aff_add):<20}")
    print(f"{'2. 동차성 T(c·u)=c·T(u)':<18} | {str(lin_hom):<20} | {str(aff_hom):<20}")
    print(f"{'3. 원점 보존 T(0)=0':<18} | {str(lin_zero):<20} | {str(aff_zero):<20}")
    print("-" * 65 + "\n")

    # 7. 실제 데이터 X2에 두 변환 적용
    X2_linear = T_linear(X2)
    X2_affine = T_affine(X2)

    # 8. 선형 변환 미성립 이유 및 딥러닝에서 XW + b를 쓰는 이유 정리
    print("=== 선형 변환 미성립 이유 및 딥러닝에서의 필요성 ===")
    print(
        "1) 선형 변환 미성립 이유: 평행이동(bias)이 포함되면 원점이 다른 위치로 이동하므로 T(0) ≠ 0이 되어 가산성과 동차성을 모두 위배하며, 엄밀하게는 '선형 변환'이 아닌 '아핀 변환(Affine Transformation)'이 됩니다."
    )
    print(
        "2) 딥러닝에서 XW + b를 사용하는 이유: 순수 선형 변환만 사용하면 모든 결정 경계가 원점을 지나야 하는 치명적 제약이 생깁니다. 편향(bias) $b$를 더해 결정 경계를 원하는 위치로 평행이동시킴으로써 모델의 표현력(Capacity)을 극대화할 수 있습니다."
    )

    # 9. 시각화 출력 (T_linear vs T_affine 비교)
    print("\n[안내] 선형 변환(T_linear)과 아핀 변환(T_affine) 산점도 창을 띄웁니다.")
    plot_pair(
        X2_linear,
        X2_affine,
        label_after="T_affine (T_linear + bias)",
        title="Linear vs Affine Transformation (Shifted by bias=[3, -2])",
    )

if __name__ == "__main__":
    main()
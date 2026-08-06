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


if __name__ == "__main__":
    main()
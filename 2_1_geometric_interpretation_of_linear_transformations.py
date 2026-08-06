import numpy as np
from dataset_2_1 import X2, apply_T, plot_pair

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    # -------------------------------------------------------------
    # [문제 1-1] 스케일링 행렬 적용하기
    # -------------------------------------------------------------
    print("=== [2장-1강] 문제 1-1 출력 결과 ===")

    # 1. x축 2배, y축 0.8배 스케일링 행렬 S 정의
    S = np.array([[2.0, 0.0], [0.0, 0.8]])

    # 2. apply_T를 활용한 선형 변환 계산
    X2_scaled = apply_T(S, X2)

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


if __name__ == "__main__":
    main()
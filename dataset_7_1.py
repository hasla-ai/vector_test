import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)

document_titles = [
    "비밀번호 재설정",
    "계정 잠금 해제",
    "VPN 오류",
    "재택근무 정책",
    "GCP GPU 신청",
    "연차 신청",
    "법인카드 정산",
    "Kubernetes 배포",
]

# 열 0: Semantic Similarity, 열 1: Keyword Overlap
X_features = np.array([
    [0.96, 0.92],
    [0.90, 0.72],
    [0.78, 0.48],
    [0.60, 0.18],
    [0.52, 0.12],
    [0.35, 0.05],
    [0.22, 0.03],
    [0.47, 0.08],
], dtype=float)

# 사람이 평가한 관련성 점수입니다.
y_human = np.array([
    0.97, 0.86, 0.72, 0.50,
    0.41, 0.24, 0.12, 0.34,
], dtype=float)

initial_w = np.array([0.2, 0.2], dtype=float)
initial_b = 0.1

print("X_features shape:", X_features.shape)
print("y_human shape:", y_human.shape)
print("initial_w:", initial_w)
print("initial_b:", initial_b)
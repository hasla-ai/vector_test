## 실제 사내 데이터가 아니라, 계산 구조를 학습하기 위해 직접 구성한 고정값

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

# 문의: "사내 비밀번호 재설정 방법"
tokens = ["<BOS>", "사내", "비밀번호", "재설정", "방법"]

# 각 토큰을 4차원 특징 벡터로 표현한 교육용 임베딩입니다.
# X.shape = (seq_len, hidden_dim) = (5, 4)
X = np.array([
    [0.2, 0.1, 0.0, 0.4],  # <BOS>
    [0.4, 0.2, 0.1, 0.6],  # 사내
    [0.9, 0.1, 0.3, 0.2],  # 비밀번호
    [0.8, 0.1, 0.9, 0.4],  # 재설정
    [0.2, 0.2, 0.6, 0.9],  # 방법
], dtype=float)

# hidden_dim=4를 d_k=3, d_v=2 공간으로 바꾸는 고정 Projection입니다.
W_Q = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.1, 0.1, 0.1],
], dtype=float)

W_K = W_Q.copy()

W_V = np.array([
    [0.4, 0.1],
    [0.1, 0.5],
    [0.3, 0.4],
    [0.2, 0.2],
], dtype=float)

print("X shape:", X.shape)
print("W_Q shape:", W_Q.shape)
print("W_K shape:", W_K.shape)
print("W_V shape:", W_V.shape)


batch_tokens = [
    ["<BOS>", "사내", "비밀번호", "재설정", "방법"],
    ["<BOS>", "GCP", "GPU", "할당", "요청"],
]

X_2 = np.array([
    [0.1, 0.1, 0.0, 0.4],
    [0.3, 0.9, 0.1, 0.3],
    [0.4, 0.8, 0.4, 0.2],
    [0.3, 0.6, 0.9, 0.3],
    [0.5, 0.5, 0.8, 0.7],
], dtype=float)

X_batch = np.stack([X, X_2], axis=0)
print("X_batch shape:", X_batch.shape)
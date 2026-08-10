import numpy as np

np.set_printoptions(precision=6, suppress=True)

# 각 행은 하나의 검색 후보 문서입니다.
# 열 0: Semantic Similarity, 열 1: Keyword Overlap
X = np.array([
    [0.95, 0.90],
    [0.82, 0.55],
    [0.48, 0.15],
    [0.20, 0.05],
], dtype=float)

y_true = np.array([
    [0.96],
    [0.78],
    [0.38],
    [0.12],
], dtype=float)

W1 = np.array([
    [0.6, -0.2],
    [0.3,  0.5],
], dtype=float)

b1 = np.array([0.05, -0.02], dtype=float)

W2 = np.array([
    [0.7],
    [0.4],
], dtype=float)

b2 = np.array([0.03], dtype=float)

print("X shape:", X.shape)
print("W1/b1 shape:", W1.shape, b1.shape)
print("W2/b2 shape:", W2.shape, b2.shape)
print("y_true shape:", y_true.shape)
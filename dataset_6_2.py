import numpy as np

np.set_printoptions(precision=4, suppress=True)

input_tokens = ["<BOS>", "사내", "비밀번호", "재설정", "방법"]
vocabulary = ["<BOS>", "사내", "비밀번호", "재설정", "방법", "안내", "<EOS>"]

X = np.array([
    [0.2, 0.1, 0.0, 0.4],
    [0.4, 0.2, 0.1, 0.6],
    [0.9, 0.1, 0.3, 0.2],
    [0.8, 0.1, 0.9, 0.4],
    [0.2, 0.2, 0.6, 0.9],
], dtype=float)

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

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Context 2차원을 Vocabulary 7개에 대한 logits로 바꾸는 출력 Projection입니다.
W_vocab = np.array([
    [0.2, -0.1, 0.3, 0.0, -0.2, 0.1, 0.05],
    [-0.3, 0.4, -0.1, 0.2, 0.1, -0.2, 0.30],
], dtype=float)

b_vocab = np.array([0.0, 0.1, -0.05, 0.0, 0.05, -0.1, 0.0], dtype=float)

# 각 입력 위치에서 맞혀야 하는 다음 토큰 index입니다.
targets = np.array([1, 2, 3, 4, 6], dtype=int)
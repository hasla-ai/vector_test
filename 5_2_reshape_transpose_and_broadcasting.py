import numpy as np
import pandas as pd
from dataset_5_2 import A_small

print("=" * 60)
print("[문제 1-1 : reshape로 배치 텐서 만들기]")
print("=" * 60)

# 1. A_small shape 및 size 확인
n_feat = A_small.shape[1]
print(f"1. A_small shape: {A_small.shape}, size: {A_small.size}")

# 2. (4, 6, 5)로 reshape
T = A_small.reshape(4, 6, n_feat)
print(f"2. T shape: {T.shape}, size: {T.size}")

# 3. -1 축 사용
T_auto = A_small.reshape(4, -1, n_feat)
print(f"3. reshape(4, -1, {n_feat}) -> shape: {T_auto.shape}")

# 4. 잘못된 reshape 예외
try:
    A_small.reshape(5, 6, n_feat)
except ValueError as e:
    print(f"4. 잘못된 reshape 시도 예외: {e}")

# 5. 첫 원소 일치 확인
print(f"5. 첫 원소 일치 여부 (A_small[0,0] == T[0,0,0]): {A_small[0, 0] == T[0, 0, 0]}")
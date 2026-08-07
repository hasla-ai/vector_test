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

print("\n" + "=" * 60)
print("[문제 1-2 : transpose와 reshape의 차이 확인하기]")
print("=" * 60)

# 1. transpose 적용
T_t = T.transpose(0, 2, 1)
print(f"1. T_t shape (transpose): {T_t.shape}")

# 2. reshape 적용
T_r = T.reshape(4, n_feat, 6)
print(f"2. T_r shape (reshape)  : {T_r.shape}")

# 3. 값 일치 여부 및 첫 구간 비교
print(f"3. T_t와 T_r 값 일치 여부: {np.array_equal(T_t, T_r)}")
print(f"   T_t[0, 0] : {np.round(T_t[0, 0], 3)}")
print(f"   T_r[0, 0] : {np.round(T_r[0, 0], 3)}")

# 4. transpose 복원
T_restored = T_t.transpose(0, 2, 1)
print(f"4. 복원 확인 (transpose 한번 더 적용): {np.array_equal(T_restored, T)}")

# 5. 차이점 정리
print("5. 정리: transpose는 축의 원래 의미와 원소 관계를 유지하며 순서만 바꾸지만, reshape는 원소를 순서대로 재배치합니다.")

print("\n" + "=" * 60)
print("[문제 2-1 : expand_dims와 squeeze로 축 조작하기]")
print("=" * 60)

sample = T[0]
print(f"1. 원본 sample shape: {sample.shape}")

# 2~3. axis 위치별 expand_dims
for ax in [0, 1, -1]:
    print(f"   expand_dims(axis={ax:>2}) -> shape: {np.expand_dims(sample, axis=ax).shape}")

# 4. squeeze 복원
batched = np.expand_dims(sample, axis=0)
restored = np.squeeze(batched, axis=0)
print(f"4. squeeze(axis=0) 후 shape: {restored.shape}, 원본 일치: {np.array_equal(restored, sample)}")

# 5. 잘못된 squeeze 시도
try:
    np.squeeze(batched, axis=1)
except ValueError as e:
    print(f"5. 크기가 1이 아닌 축 squeeze 실패 예외: {e}")

# 6. 설명
print("6. 이유: 축 개수(ndim)가 다르면 프레임워크에서 다른 구조의 텐서로 인식하여 차원 불일치 에러를 발생시킵니다.")

print("\n" + "=" * 60)
print("[문제 2-2 : broadcasting으로 센서별 표준화 직접 구현하기]")
print("=" * 60)

# 1. mean, std (ddof=0)
mean = A_small.mean(axis=0)
std = A_small.std(axis=0) + 1e-8
print(f"1. A_small: {A_small.shape}, mean: {mean.shape}, std: {std.shape}")

# 2. broadcasting 표준화
A_scaled = (A_small - mean) / std
print(f"2. A_scaled shape: {A_scaled.shape}")

# 3. 평균 0, 표준편차 1 검증
print(f"3. 센서별 평균(0에 수렴): {np.round(A_scaled.mean(axis=0), 6)}")
print(f"   센서별 표준편차(1에 수렴): {np.round(A_scaled.std(axis=0), 6)}")

# 4. (1, 5) 명시적 reshape 비교
mean_2d = mean.reshape(1, -1)
print(f"4. 명시적 (1, 5) 사용 결과와 동일 여부: {np.allclose((A_small - mean_2d) / std, A_scaled)}")

# 5. 잘못된 방향 빼기 (keepdims 미지정 vs 지정)
try:
    A_small - A_small.mean(axis=1)
except ValueError as e:
    print(f"5-1. keepdims 없이 axis=1 연산 시 오류: {e}")

row_mean = A_small.mean(axis=1, keepdims=True)
print(f"5-2. keepdims=True (24, 1) 연산 결과 shape: {(A_small - row_mean).shape}")
print(f"     잘못 적용 후 센서별 평균: {np.round((A_small - row_mean).mean(axis=0)[:3], 3)}")

# 6. 규칙 정리
print("6. broadcasting 규칙: 뒤쪽 축부터 비교하여 크기가 같거나 한쪽 축 크기가 1이면 자동으로 축을 늘려서 연산합니다.")
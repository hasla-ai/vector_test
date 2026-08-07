# 5_1_tensor_and_shape.py
import numpy as np
import pandas as pd
from dataset_5_1 import get_processed_seqs

# ----------------------------------------------------
# 필수 1 : 딥러닝 코드에 나오는 숫자 뭉치 읽기
# ----------------------------------------------------
print("=== [문제 1-1] 벡터·행렬·텐서의 shape 읽기 ===")
vector = np.array([3, 5, 7])
matrix = np.array([[1, 2], [3, 4], [5, 6]])
tensor3d = np.zeros((2, 3, 4))

for name, arr in [('벡터', vector), ('행렬', matrix), ('텐서', tensor3d)]:
    print(f'{name:4s}: shape={str(arr.shape):10s}, ndim={arr.ndim}, size={arr.size}')

x = np.random.randn(32, 10, 512)  # (batch, seq_len, hidden_dim)
print(f'\nx.shape = {x.shape}')
print('해석: 문장 32개를 한 번에 처리하고(batch), 각 문장은 단어 10개(seq_len), '
      '각 단어는 512차원 벡터(hidden_dim)로 표현됩니다.')

print('\n[인덱싱 Shape 예측 및 실행]')
print('x[0]   예측: (10, 512)  | 실제:', x[0].shape, ' -> 문장 1개 (seq_len, hidden_dim)')
print('x[0, 0] 예측: (512,)     | 실제:', x[0, 0].shape, '-> 단어 1개의 임베딩 벡터')


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

print("\n=== [문제 1-2] 이미지 텐서 (batch, channel, height, width) 해석하기 ===")
images = np.random.randn(16, 3, 224, 224)     # PyTorch NCHW
gray = np.random.randn(16, 1, 224, 224)       # 흑백은 채널 1

print('컬러 이미지 배치 (NCHW):', images.shape)
print('흑백 이미지 배치 (NCHW):', gray.shape)
print('이미지 1장           :', images[0].shape, '-> (channel, height, width)')
print('1장의 R 채널         :', images[0, 0].shape, '-> (height, width)')

images_nhwc = images.transpose(0, 2, 3, 1)    # TensorFlow NHWC 변환
print('\nNCHW -> NHWC 변환:', images.shape, '->', images_nhwc.shape)

print('\n[프레임워크별 이미지 텐서 표기 비교]')
df_img_compare = pd.DataFrame({
    '프레임워크': ['PyTorch', 'TensorFlow (기본)'],
    '축 순서': ['(batch, channel, height, width)', '(batch, height, width, channel)'],
    '명칭': ['NCHW', 'NHWC']
})
print(df_img_compare.to_string(index=False))
print('\n축 순서 오해 시 발생 문제:')
print(' -> 에러 메시지 없이 224 채널이나 3 높이/너비로 잘못 연산되어 모델 성능이 왜곡되거나 silent failure가 발생합니다.')

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


# ----------------------------------------------------
# 필수 2 : 길이가 제각각인 고객 이력을 하나의 텐서로 묶기
# ----------------------------------------------------
print("\n=== [문제 2-1] padding으로 시퀀스 길이 맞추기 ===")
df, seqs = get_processed_seqs()
lengths = [len(s) for s in seqs]
print('고객별 시퀀스 길이:', lengths)

# ① 그대로 묶었을 때 ValueError 발생 확인
try:
    raw = np.array(seqs.tolist())
    print('① 그대로 묶었을 때 shape:', raw.shape)
except ValueError as e:
    print('① np.array() 그대로 -> ValueError 발생:', e)

# ② dtype=object 명시 시 1차원 object 배열 확인
raw_obj = np.array(seqs.tolist(), dtype=object)
print('② dtype=object 지정  -> shape:', raw_obj.shape,
      '/ dtype:', raw_obj.dtype, '/ ndim:', raw_obj.ndim)

# ③ 정수 인코딩 및 데이터 기반 max_len padding 처리
items = sorted(df['StockCode'].unique())
item_to_id = {it: i + 1 for i, it in enumerate(items)}   # 0은 padding 전용 ID

max_len = max(lengths)   # 실제 데이터의 최대 길이로 동적 계산
encoded, mask = [], []
for s in seqs:
    ids = [item_to_id[x] for x in s[:max_len]]
    pad = max_len - len(ids)
    encoded.append(ids + [0] * pad)
    mask.append([1] * len(ids) + [0] * pad)

encoded, mask = np.array(encoded), np.array(mask)
print(f'\n계산된 max_len: {max_len}')
print('encoded shape:', encoded.shape, '-> (batch, seq_len)')
print('mask shape   :', mask.shape)
print('\n[encoded 앞 2행]:\n', encoded[:2])
print('[mask 앞 2행]:\n', mask[:2])
print('\nmask의 필요성:')
print(' -> padding 처리된 0번 토큰은 실제 거래가 아니므로, 손실(Loss) 계산이나 어텐션 연산 시 모델이 이를 학습하지 않도록 차단하기 위해 필요합니다.')

print("\n=== [문제 2-2] 임베딩으로 3차원 텐서 만들기 ===")
hidden_dim = 16
E = np.random.randn(len(item_to_id) + 1, hidden_dim)
E[0] = 0  # 0번(padding) 임베딩 벡터는 0으로 처리

X_tensor = E[encoded]
print('임베딩 테이블 E shape:', E.shape)
print('결과 텐서 shape       :', X_tensor.shape, '-> (batch, seq_len, hidden_dim)')
print(f'축 해석: 고객 {X_tensor.shape[0]}명을 한 번에 처리하고(batch), '
      f'각 고객은 상품 {X_tensor.shape[1]}개(seq_len), '
      f'각 상품은 {X_tensor.shape[2]}차원 벡터(hidden_dim)로 표현됩니다.')

pad_pos = (mask == 0)
print('\npadding 위치의 임베딩 벡터가 모두 0인가?:', np.allclose(X_tensor[pad_pos], 0))

# hidden_dim = 32 변경 검증
E32 = np.random.randn(len(item_to_id) + 1, 32)
E32[0] = 0
X_tensor_32 = E32[encoded]
print('hidden_dim을 32로 변경 시 shape 예측: (8, 8, 32) | 실제:', X_tensor_32.shape)

# ----------------------------------------------------
# 심화 1 : 실행하기 전에 연산 가능 여부 판단하기
# ----------------------------------------------------
print("\n=== [문제 3-1] shape 조합별 연산 가능 여부 확인하기 ===")
a = np.random.randn(32, 10, 512)
b = np.random.randn(32, 10, 512)
c = np.random.randn(32, 5, 512)
d = np.random.randn(512)

print(f'a.shape={a.shape}, b.shape={b.shape}, c.shape={c.shape}, d.shape={d.shape}\n')

operations = [('a + b', lambda: a + b, '가능', '(32, 10, 512)'),
              ('a + c', lambda: a + c, '불가능', 'ValueError'),
              ('a + d', lambda: a + d, '가능', '(32, 10, 512)')]

results = []
for op_name, op_func, pred_status, pred_shape in operations:
    try:
        res = op_func()
        exec_status = '성공'
        exec_res = str(res.shape)
    except ValueError as err:
        exec_status = '실패'
        exec_res = f'ValueError ({err})'
    
    results.append({
        '연산 조합': op_name,
        '예측 가능 여부': pred_status,
        '예측 Result/Error': pred_shape,
        '실제 실행 결과': exec_status,
        '실제 Result/Error': exec_res
    })

df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))

print('\n[해설 및 질문 답변]')
print('1. a + c를 가능하게 하려면?:')
print(' -> 시퀀스 차원(축 1)의 크기가 10과 5로 다르므로, c의 시퀀스를 padding하여 (32, 10, 512)로 맞추거나 슬라이싱 a[:, :5, :]을 적용해야 합니다.')

print('2. 브로드캐스팅 축 비교 규칙:')
print(' -> 두 텐서의 shape을 "뒤쪽 축(마지막 차원)"부터 비교할 때, 각 차원의 크기가 동일하거나 어느 한쪽이 1인 경우에만 연산(브로드캐스팅)이 가능합니다.')
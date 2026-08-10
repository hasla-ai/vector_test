## ** [7장 2강] - 실습: Chain Rule과 Backpropagation ** ##

# 필수 1. 한 요청의 계산 그래프를 따라가기

## 문제 1-1: Forward 값과 Upstream Gradient 기록

### 업무 요청

먼저 첫 번째 문서 한 건만 사용해 계산 흐름을 추적하세요. 첫 표본의 계산은 다음과 같습니다.

x = X[0]
h = x @ W1 + b1
prediction = h @ W2 + b2
loss = (prediction - target)²

### 수행해야 할 작업

1. 첫 표본의 `h`, `prediction`, `loss`를 계산하세요.
2. `dLoss/dPrediction`을 계산하세요.
3. `dPrediction/dh = W2`를 이용해 `dLoss/dh`를 계산하세요.
4. 다음 표를 작성하세요.

node | forward value | local gradient | upstream gradient | output gradient

“경로 안에서는 곱한다”는 말을 이 계산에 맞게 설명하세요.

시작 코드

```bash
def trace_single_request(x, target, W1, b1, W2, b2):
    """한 표본의 Forward와 핵심 Backward 값을 반환하세요."""
    #TODO
    raise NotImplementedError
```

힌트

```
loss = (prediction - target)²
dLoss/dPrediction = 2(prediction - target)
dPrediction/dh = W₂
```

prediction과 target이 Shape (1,) 배열이라면 .item()으로 스칼라를 꺼낼 수 있습니다.

결과

```bash
X shape: (4, 2)
W1/b1 shape: (2, 2) (2,)
W2/b2 shape: (2, 1) (1,)
y_true shape: (4, 1)
======================================================================
[필수 1-1: 첫 번째 표본 계산 그래프 추적]
======================================================================
h: [0.89 0.24]
prediction: 0.749
loss: 0.044520999999999984
dLoss/dPrediction: -0.42199999999999993
dPrediction/dh: [0.7 0.4]
dLoss/dh: [-0.2954 -0.1688]

node | forward value | local gradient | upstream gradient | output gradient
MSE  | prediction=0.7490 | 2(pred-target)=-0.4220 | 1.0000 | -0.4220
W2   | h=[0.89 0.24] | W2=[0.7 0.4] | -0.4220 | [-0.2954 -0.1688]
```

h → prediction → loss 경로를 거꾸로 이동할 때, 뒤에서 전달받은 dLoss/dPrediction과 현재 연산의 Local Gradient인 dPrediction/dh를 곱하여 dLoss/dh를 구합니다. 이것이 “경로 안에서는 Gradient를 곱한다”는 의미입니다.

# 필수 2. 두 단계 모델의 수동 Backpropagation

## ▶ 문제 2-1: 배치 Gradient 계산과 한 번의 업데이트

### 업무 요청

이제 네 표본 전체를 사용해 `W1`, `b1`, `W2`, `b2`의 Gradient를 계산하세요. 반복문으로 원소를 하나씩 미분하기보다 행렬곱으로 전체 배치를 처리합니다.

## Forward ##
H = X @ W1 + b1
prediction = H @ W2 + b2
Loss = mean((prediction - y_true)²)

## Backward ##
dPrediction = 2(prediction - y_true) / N

dW2 = Hᵀ @ dPrediction
db2 = sum(dPrediction)

dH = dPrediction @ W2ᵀ

dW1 = Xᵀ @ dH
db1 = sum(dH, axis=0)

### 수행해야 할 작업

1. `forward_backward()` 함수 전체를 구현하세요.
2. 모든 중간 배열의 Shape을 출력하세요.
3. Learning Rate `0.2`로 네 파라미터를 한 번 업데이트하세요.
4. 업데이트 전후 Loss를 비교하세요.
5. 다음 조건을 확인하세요.
    - `dW1.shape == W1.shape`
    - `db1.shape == b1.shape`
    - `dW2.shape == W2.shape`
    - `db2.shape == b2.shape`
    - 업데이트 후 Loss가 감소함

시작 코드

```bash
def forward_backward(X, y_true, W1, b1, W2, b2):
    """Forward 값과 모든 파라미터 Gradient를 반환하세요."""
    #TODO
    raise NotImplementedError
```

### 제출해야 할 보고 형식

```
[Backpropagation 검증 보고]
- 업데이트 전 Loss:
- 업데이트 후 Loss:
- dW1 Shape:
- db1 Shape:
- dW2 Shape:
- db2 Shape:
- Loss 감소 여부:
- Backpropagation과 Gradient Descent의 역할 차이:
```

- 힌트 보기
    - `N`은 표본 수 `X.shape[0]`입니다.
    - `db1`은 배치 축을 합치므로 `axis=0`입니다.
    - 파라미터 업데이트는 Backpropagation이 아니라 Gradient Descent의 역할입니다.

결과

```bash
======================================================================
[필수 2-1: 전체 배치 Backpropagation 및 파라미터 업데이트]
======================================================================
H shape: (4, 2)
prediction shape: (4, 1)
dPrediction shape: (4, 1)
dH shape: (4, 2)
dW1 shape: (2, 2)
db1 shape: (2,)
dW2 shape: (2, 1)
db2 shape: (1,)

[Backpropagation 검증 보고]
- 업데이트 전 Loss: 0.025666
- 업데이트 후 Loss: 0.005661
- dW1 Shape: (2, 2)
- db1 Shape: (2,)
- dW2 Shape: (2, 1)
- db2 Shape: (1,)
- Loss 감소 여부: True
- Backpropagation과 Gradient Descent의 역할 차이:
  * Backpropagation: 연쇄 법칙(Chain Rule)을 이용하여 각 파라미터별 Gradient를 효율적으로 계산합니다.
  * Gradient Descent: 계산된 Gradient 방향의 반대로 파라미터를 업데이트하여 Loss를 감소시킵니다.
```

dPrediction → dH로 이동할 때 W2.T를 곱하고, dH에서 dW1을 구할 때 X.T를 곱합니다. 각 행렬곱의 Shape이 맞아야 Gradient가 원래 파라미터와 같은 Shape을 갖습니다.

# 심화 1. PyTorch Autograd와 수동 Gradient 비교

## ▶ 문제 3-1: `backward()` 결과와 `zero_grad()` 확인

### 업무 상황

수동 Backpropagation이 맞는지 PyTorch Autograd로 교차검증합니다. 또한 PyTorch가 Gradient를 기본적으로 **누적**한다는 점을 확인합니다.

### 수행해야 할 작업

1. NumPy 데이터를 `torch.float64` Tensor로 변환하세요.
2. 네 파라미터에 `requires_grad=True`를 설정하세요.
3. 같은 Forward와 MSE를 구현하고 `loss.backward()`를 호출하세요.
4. 수동 Gradient와 최대 절대 차이를 비교하세요.
5. `zero_grad()` 없이 같은 Forward/Backward를 한 번 더 수행해 Gradient가 누적되는지 확인하세요.
6. `zero_grad()`를 호출한 뒤 Gradient가 초기화되는지 확인하세요.

힌트 보기
    - 두 번째 `backward()`를 호출하기 전에 **Forward를 다시 계산**하면 새 계산 그래프가 만들어집니다.
    - `tensor.grad.zero_()` 또는 Optimizer의 `zero_grad()`로 Gradient를 초기화할 수 있습니다.
    - NumPy 비교 전에는 `.detach().numpy()`를 사용하세요.



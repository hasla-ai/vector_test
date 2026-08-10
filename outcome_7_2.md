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


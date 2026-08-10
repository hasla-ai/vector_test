## ** [7장 1강] - 실습: Gradient Descent ** ##


# 필수 1. 해석적 Gradient가 맞는지 검증하기

## ▶ 문제 1-1: Gradient Check 수행

### 업무 요청

학습 코드를 배포하기 전에 Gradient 식이 맞는지 검증해야 합니다. 잘못된 Gradient는 코드가 실행되더라도 Loss를 증가시키거나 수렴을 방해할 수 있습니다.

모델과 MSE는 다음과 같습니다.

y_pred = X @ w + b
Loss = mean((y_pred - y_human)²)

Gradient는 다음 구조를 가집니다.

grad_w = 2/N × Xᵀ @ (y_pred - y_human)
grad_b = 2/N × sum(y_pred - y_human)

### 수행해야 할 작업

1. `predict_relevance()`를 구현하세요.
2. `mse_loss()`를 구현하세요.
3. `mse_gradients()`에서 `grad_w`, `grad_b`를 계산하세요.
4. 중앙차분 기반 `numerical_parameter_gradient()`로 `w[0]`, `w[1]`, `b`를 각각 검증하세요.
5. 해석적·수치 Gradient의 최대 절대 차이를 출력하세요.
6. 현재 Gradient의 부호를 보고 각 파라미터가 첫 업데이트에서 증가할지 감소할지 설명하세요.

시작 코드

```bash
def predict_relevance(X, w, b):
    #TODO
    raise NotImplementedError


def mse_loss(y_pred, y_true):
    #TODO
    raise NotImplementedError


def mse_gradients(X, y_true, w, b):
    #TODO: y_pred, error, grad_w, grad_b를 계산하세요.
    raise NotImplementedError


def numerical_parameter_gradient(X, y_true, w, b, parameter_index, h=1e-5):
    """
    parameter_index가 0 또는 1이면 w의 해당 원소를,
    2이면 b를 중앙차분으로 미분하세요.
    """
    #TODO
    raise NotImplementedError
```

제출해야 할 결과

analytic grad_w:
analytic grad_b:
numerical gradients:
max absolute difference:

그리고 다음 문장을 완성하세요.

[Gradient 검증 보고]
- 최대 절대 차이:
- Gradient 구현 정상 여부:
- w₁ 업데이트 방향:
- w₂ 업데이트 방향:
- b 업데이트 방향:

- 힌트
    - 중앙차분: `(L(θ+h) - L(θ-h)) / (2h)`
    - `w.copy()`로 원본 파라미터를 보호하세요.
    - Gradient Descent는 `parameter -= learning_rate * gradient`입니다.
    - Gradient가 음수이면 빼기 연산 때문에 파라미터가 증가합니다.

결과

```bash
X_features shape: (8, 2)
y_human shape: (8,)
initial_w: [0.2 0.2]
initial_b: 0.1
============================================================
[Gradient Check 계산 결과]
============================================================
analytic grad_w:         [-0.36755 -0.25583]
analytic grad_b:         -0.47100
numerical gradients:     [-0.36756 -0.25582 -0.471  ]
max absolute difference: 8.89e-13

============================================================
[Gradient 검증 보고]
============================================================
- 최대 절대 차이: 8.89e-13
- Gradient 구현 정상 여부: 정상 (차이가 1e-5 미만으로 극히 미미함)
- w₁ 업데이트 방향: 증가 (Gradient가 음수이므로 w -= lr * grad에 의해 값 증가)
- w₂ 업데이트 방향: 증가 (Gradient가 음수이므로 w -= lr * grad에 의해 값 증가)
- b 업데이트 방향: 증가 (Gradient가 음수이므로 w -= lr * grad에 의해 값 증가)
============================================================
```

  세 Gradient가 모두 음수이므로 parameter -= lr × gradient를 적용하면 첫 단계에서 w₁, w₂, b가 모두 증가합니다. 이 검증은 해석적 식과 코드 구현이 같은 변화율을 계산하는지 확인하는 과정입니다.





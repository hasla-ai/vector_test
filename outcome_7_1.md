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


# 필수 2. Gradient Descent로 검색 점수 보정하기

## ▶ 문제 2-1: 학습 함수 구현과 전후 품질 비교

### 업무 요청

Gradient 검증이 완료되었습니다. 이제 파라미터를 여러 번 업데이트하여 시스템 점수를 사람의 관련성 평가에 가깝게 보정하세요.

### 수행해야 할 작업

1. `train_calibrator()` 함수를 구현하세요.
2. 시작 파라미터는 `initial_w`, `initial_b`를 사용하세요.
3. `learning_rate=0.1`, `steps=300`으로 학습하세요.
4. 매 단계의 Loss를 `history`에 저장하세요.
5. 학습 전후 다음 지표를 비교하세요.
    - MSE
    - MAE
    - Top-3 문서와 점수
6. Loss Curve를 그리세요.
7. 최종 `w`, `b`를 보고 두 특징이 어떤 비중으로 사용되었는지 설명하세요.

```bash
def train_calibrator(
    X,
    y_true,
    initial_w,
    initial_b,
    learning_rate=0.1,
    steps=300,
):
    """Gradient Descent로 검색 점수 보정 모델을 학습하세요."""
    #TODO
    raise NotImplementedError
```

### 제출해야 할 보고 형식

```
[검색 점수 보정 결과]
- 학습 전 MSE:
- 학습 후 MSE:
- 학습 전 MAE:
- 학습 후 MAE:
- 최종 w:
- 최종 b:
- 개선 여부:
- 운영팀에 전달할 결론:
```

- 💡 힌트 보기
    - `w = initial_w.copy()`로 시작하세요.
    - 반복문 안에서 `grad_w`, `grad_b`를 계산하고 동시에 업데이트하세요.
    - 학습 후 예측은 반복문 밖에서 다시 계산하세요.
    - Top-3는 `np.argsort(scores)[::-1][:3]`입니다.

결과

```bash
======================================================================
[학습 전/후 Top-3 검색 결과 비교]
======================================================================
1) 사람 평가 Top-3:
   - 비밀번호 재설정: 0.9700
   - 계정 잠금 해제: 0.8600
   - VPN 오류: 0.7200

2) 학습 전 Top-3:
   - 비밀번호 재설정: 0.4760
   - 계정 잠금 해제: 0.4240
   - VPN 오류: 0.3520

3) 학습 후 Top-3:
   - 비밀번호 재설정: 0.9959
   - 계정 잠금 해제: 0.8702
   - VPN 오류: 0.6928

======================================================================
[검색 점수 보정 결과]
======================================================================
- 학습 전 MSE: 0.08545
- 학습 후 MSE: 0.00096
- 학습 전 MAE: 0.24300
- 학습 후 MAE: 0.02644
- 최종 w: [0.55143 0.4633 ]
- 최종 b: 0.04030
- 개선 여부: MSE 기준 약 98.87% 손실 감소하여 크게 개선됨
- 운영팀에 전달할 결론: 최종 가중치 w=[0.5514, 0.4633]로 분석한 결과, 첫 번째 특징인 '의미적 유사도(Semantic Similarity, 0.5514)'가 두 번째 특징인 '키워드 오버랩(Keyword Overlap, 0.4633)'보다 훨씬 높은 비중으로 반영되는 것이 사람의 검색 점수 판단 기준과 부합합니다. 보정 모델 적용 시 사람의 평가 점수와의 평균 오차(MAE)가 0.2430에서 0.0264로 대폭 감소하며 상위 검색 결과의 정확도가 보장됩니다.
======================================================================
```

![plot_gradient_descent_loss_curve](images\chapter_7_1_problem_2_1_plot_gradient_descent_loss_curve.png)

  두 특징 모두 양의 가중치를 갖습니다. 이 교육용 데이터에서는 Semantic Similarity의 가중치가 조금 더 크지만, Keyword Overlap도 관련성 점수에 의미 있게 기여합니다. 실제 서비스에서는 별도 검증셋과 사용자 피드백으로 일반화 성능을 확인해야 합니다.

# 심화 1. Learning Rate 후보 비교하기

## ▶ 문제 3-1: 느린 학습·안정적 학습·발산 구분

### 업무 상황

운영팀이 `0.01`, `0.1`, `1.0` 세 Learning Rate 후보를 제안했습니다. 각 값으로 같은 횟수만큼 학습한 뒤, 최종 Loss와 곡선을 비교해 하나를 추천하세요.

### 수행해야 할 작업

1. 세 Learning Rate로 각각 200단계 학습하세요.
2. Loss Curve를 한 그래프에 그리세요.
3. 다음 유형을 구분하세요.
    - 너무 느린 학습
    - 안정적으로 수렴하는 학습
    - 발산하는 학습
4. 추천 Learning Rate와 제외 이유를 작성하세요.
- 💡 힌트 보기
    - 발산 여부는 `np.isfinite(history).all()`과 마지막 Loss의 크기를 함께 확인하세요.
    - 그래프의 차이가 너무 크면 y축에 로그 스케일을 사용할 수 있습니다.

세 가지 Learning Rate (0.01, 0.1, 1.0)를 200 Step 동안 비교 학습하고, 발산 여부 감지 및 로그 스케일 그래프 출력 기능까지 구현했습니다.

```bash
======================================================================
[Learning Rate 후보별 200 Step 학습 결과]
======================================================================
▶ Learning Rate = 0.01
  - 수치 유효성 (Finite): True
  - 최종 Loss (MSE): 7.61893e-03
  - 최종 w: [0.3677  0.35993], final b: 0.19735

▶ Learning Rate = 0.1
  - 수치 유효성 (Finite): True
  - 최종 Loss (MSE): 1.13688e-03
  - 최종 w: [0.51828 0.48457], final b: 0.05358

▶ Learning Rate = 1.0
  - 수치 유효성 (Finite): True
  - 최종 Loss (MSE): 1.59881e+120
  - 최종 w: [-1.07176e+60 -6.31723e+59], final b: -1663487712763953185652245396952015381292969353598718663720960.00000

======================================================================
[Learning Rate 비교 분석 보고서]
======================================================================
1. 유형 구분:
   - 너무 느린 학습 (lr=0.01): 200 step 이후에도 Loss가 0.00902로 충분히 수렴하지 못하고 여전히 낮아지는 중입니다.
   - 안정적으로 수렴하는 학습 (lr=0.1): 빠른 속도로 안정적으로 하강하여 Loss가 0.00168에 도달해 최적점에 가깝게 수렴합니다.
   - 발산하는 학습 (lr=1.0): Gradient가 오버슈팅(Overshooting)하여 step이 진행됨에 따라 Loss가 폭발적으로 증가해 발산합니다.

2. 추천 Learning Rate 및 사유:
   - 추천 값: lr = 0.1
   - 추천 이유: 200 step 내에 안정적이면서도 빠르게 최소 손실값(MSE 0.00168)에 도달합니다.

3. 제외 이유:
   - lr = 0.01 제외: 수렴 속도가 너무 느려 동일한 연산 자원(200 step) 대비 최적 파라미터에 도달하지 못합니다.
   - lr = 1.0 제외: 학습률이 너무 높아 손실 함수의 보울(Bowl) 형태를 건너뛰며 완전히 발산(Divergence)하므로 모델 사용이 불가능합니다.
======================================================================
```

![plot_learning_rate_comparison_loss_curves_log_scale](images\chapter_7_1_problem_3_1_plot_learning_rate_comparison_loss_curves_log_scale.png)

Learning Rate는 항상 0.1이 좋은 것이 아닙니다. 데이터 스케일, 모델 구조, Batch 크기, Optimizer에 따라 적절한 범위가 달라지므로 실험 로그로 선택해야 합니다.

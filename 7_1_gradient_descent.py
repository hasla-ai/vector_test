import numpy as np
from dataset_7_1 import document_titles, X_features, y_human, initial_b, initial_w

# 필수 1. 해석적 Gradient가 맞는지 검증하기
## ▶ 문제 1-1: Gradient Check 수행

# ----------------------------------------------------
# 2. 필수 함수 구현
# ----------------------------------------------------
def predict_relevance(X, w, b):
    """선형 예측 모델: y_pred = X @ w + b"""
    return X @ w + b


def mse_loss(y_pred, y_true):
    """평균 제곱 오차 (MSE Loss)"""
    return np.mean((y_pred - y_true) ** 2)


def mse_gradients(X, y_true, w, b):
    """해석적(Analytic) Gradient 계산"""
    N = len(y_true)
    y_pred = predict_relevance(X, w, b)
    error = y_pred - y_true

    grad_w = (2 / N) * (X.T @ error)
    grad_b = (2 / N) * np.sum(error)

    return grad_w, grad_b


def numerical_parameter_gradient(X, y_true, w, b, parameter_index, h=1e-5):
    """
    중앙차분 기반 수치 미분: (L(θ+h) - L(θ-h)) / (2h)
    parameter_index: 0 -> w[0], 1 -> w[1], 2 -> b
    """
    w_plus = w.copy()
    w_minus = w.copy()
    b_plus = b
    b_minus = b

    if parameter_index == 0:
        w_plus[0] += h
        w_minus[0] -= h
    elif parameter_index == 1:
        w_plus[1] += h
        w_minus[1] -= h
    elif parameter_index == 2:
        b_plus += h
        b_minus -= h

    pred_plus = predict_relevance(X, w_plus, b_plus)
    loss_plus = mse_loss(pred_plus, y_true)

    pred_minus = predict_relevance(X, w_minus, b_minus)
    loss_minus = mse_loss(pred_minus, y_true)

    return (loss_plus - loss_minus) / (2 * h)


# ----------------------------------------------------
# 3. Gradient Check 및 보고서 출력
# ----------------------------------------------------
# 1) 해석적 Gradient 계산
analytic_grad_w, analytic_grad_b = mse_gradients(
    X_features, y_human, initial_w, initial_b
)
analytic_grads = np.array(
    [analytic_grad_w[0], analytic_grad_w[1], analytic_grad_b]
)

# 2) 수치적 Gradient 계산 (중앙차분)
num_grad_w0 = numerical_parameter_gradient(
    X_features, y_human, initial_w, initial_b, parameter_index=0
)
num_grad_w1 = numerical_parameter_gradient(
    X_features, y_human, initial_w, initial_b, parameter_index=1
)
num_grad_b = numerical_parameter_gradient(
    X_features, y_human, initial_w, initial_b, parameter_index=2
)
numerical_grads = np.array([num_grad_w0, num_grad_w1, num_grad_b])

# 3) 최대 절대 차이 계산
abs_diff = np.abs(analytic_grads - numerical_grads)
max_abs_diff = np.max(abs_diff)

print("=" * 60)
print("[Gradient Check 계산 결과]")
print("=" * 60)
print(f"analytic grad_w:         {analytic_grad_w}")
print(f"analytic grad_b:         {analytic_grad_b:.5f}")
print(f"numerical gradients:     {numerical_grads}")
print(f"max absolute difference: {max_abs_diff:.2e}")

# 각 파라미터 업데이트 방향 결정 함수 (gradient < 0 이면 증가, > 0 이면 감소)
def get_update_direction(grad):
    return "증가 (Gradient가 음수이므로 w -= lr * grad에 의해 값 증가)" if grad < 0 else "감소"

print("\n" + "=" * 60)
print("[Gradient 검증 보고]")
print("=" * 60)
print(f"- 최대 절대 차이: {max_abs_diff:.2e}")
print(
    f"- Gradient 구현 정상 여부: {'정상 (차이가 1e-5 미만으로 극히 미미함)' if max_abs_diff < 1e-5 else '비정상'}"
)
print(f"- w₁ 업데이트 방향: {get_update_direction(analytic_grad_w[0])}")
print(f"- w₂ 업데이트 방향: {get_update_direction(analytic_grad_w[1])}")
print(f"- b 업데이트 방향: {get_update_direction(analytic_grad_b)}")
print("=" * 60)
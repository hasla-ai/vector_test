import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_7_1 import document_titles, X_features, y_human, initial_b, initial_w

np.set_printoptions(precision=5, suppress=True)

# 필수 1. 해석적 Gradient가 맞는지 검증하기
## ▶ 문제 1-1: Gradient Check 수행

# ----------------------------------------------------
# 필수 함수 구현
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
# Gradient Check 및 보고서 출력
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


## 필수 2. Gradient Descent로 검색 점수 보정하기
## ▶ 문제 2-1: 학습 함수 구현과 전후 품질 비교

# ----------------------------------------------------
# 필수 함수 및 학습 함수 구현
# ----------------------------------------------------
def predict_relevance(X, w, b):
    return X @ w + b


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def mae_loss(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))


def mse_gradients(X, y_true, w, b):
    N = len(y_true)
    y_pred = predict_relevance(X, w, b)
    error = y_pred - y_true
    grad_w = (2 / N) * (X.T @ error)
    grad_b = (2 / N) * np.sum(error)
    return grad_w, grad_b


def train_calibrator(
    X,
    y_true,
    initial_w,
    initial_b,
    learning_rate=0.1,
    steps=300,
):
    """Gradient Descent로 검색 점수 보정 모델을 학습하세요."""
    w = initial_w.copy()
    b = initial_b
    history = []

    for step in range(steps):
        # 1. 현재 예측 및 Loss 계산
        y_pred = predict_relevance(X, w, b)
        loss = mse_loss(y_pred, y_true)
        history.append(loss)

        # 2. Gradient 계산 및 파라미터 동시 업데이트
        grad_w, grad_b = mse_gradients(X, y_true, w, b)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return w, b, history

# ----------------------------------------------------
# 학습 전후 품질 비교 및 결과 출력
# ----------------------------------------------------

# 1) 학습 전 상태 지표 계산
y_pred_before = predict_relevance(X_features, initial_w, initial_b)
mse_before = mse_loss(y_pred_before, y_human)
mae_before = mae_loss(y_pred_before, y_human)

# 2) Gradient Descent 학습 실행 (lr=0.1, steps=300)
final_w, final_b, loss_history = train_calibrator(
    X_features,
    y_human,
    initial_w,
    initial_b,
    learning_rate=0.1,
    steps=300,
)

# 3) 학습 후 상태 지표 계산
y_pred_after = predict_relevance(X_features, final_w, final_b)
mse_after = mse_loss(y_pred_after, y_human)
mae_after = mae_loss(y_pred_after, y_human)

# 4) Top-3 문서 및 점수 추출
top3_before_idx = np.argsort(y_pred_before)[::-1][:3]
top3_after_idx = np.argsort(y_pred_after)[::-1][:3]
top3_human_idx = np.argsort(y_human)[::-1][:3]

print("=" * 70)
print("[학습 전/후 Top-3 검색 결과 비교]")
print("=" * 70)
print("1) 사람 평가 Top-3:")
for idx in top3_human_idx:
    print(f"   - {document_titles[idx]}: {y_human[idx]:.4f}")

print("\n2) 학습 전 Top-3:")
for idx in top3_before_idx:
    print(f"   - {document_titles[idx]}: {y_pred_before[idx]:.4f}")

print("\n3) 학습 후 Top-3:")
for idx in top3_after_idx:
    print(f"   - {document_titles[idx]}: {y_pred_after[idx]:.4f}")

# ----------------------------------------------------
# 제출용 보고서 양식 출력
# ----------------------------------------------------
print("\n" + "=" * 70)
print("[검색 점수 보정 결과]")
print("=" * 70)
print(f"- 학습 전 MSE: {mse_before:.5f}")
print(f"- 학습 후 MSE: {mse_after:.5f}")
print(f"- 학습 전 MAE: {mae_before:.5f}")
print(f"- 학습 후 MAE: {mae_after:.5f}")
print(f"- 최종 w: {final_w}")
print(f"- 최종 b: {final_b:.5f}")
print(f"- 개선 여부: MSE 기준 약 {(1 - mse_after / mse_before) * 100:.2f}% 손실 감소하여 크게 개선됨")
print(
    f"- 운영팀에 전달할 결론: 최종 가중치 w=[{final_w[0]:.4f}, {final_w[1]:.4f}]로 분석한 결과, "
    f"첫 번째 특징인 '의미적 유사도(Semantic Similarity, {final_w[0]:.4f})'가 두 번째 특징인 "
    f"'키워드 오버랩(Keyword Overlap, {final_w[1]:.4f})'보다 훨씬 높은 비중으로 반영되는 것이 사람의 검색 점수 판단 기준과 부합합니다. "
    f"보정 모델 적용 시 사람의 평가 점수와의 평균 오차(MAE)가 {mae_before:.4f}에서 {mae_after:.4f}로 대폭 감소하며 상위 검색 결과의 정확도가 보장됩니다."
)
print("=" * 70)

# ----------------------------------------------------
# Loss Curve 시각화
# ----------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Training Loss (MSE)", color="blue", linewidth=2)
plt.title("Gradient Descent Loss Curve (300 Steps)", fontsize=14)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
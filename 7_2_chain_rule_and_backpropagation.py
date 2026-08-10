import numpy as np
import torch
from dataset_7_2 import X, y_true, W1, b1, W2, b2

# ----------------------------------------------------
# 필수 1. 한 요청의 계산 그래프를 따라가기 (문제 1-1)
# ----------------------------------------------------
def trace_single_request(x, target, W1, b1, W2, b2):
    """한 표본이 두 선형변환과 MSE를 지나는 값을 추적합니다."""
    h = x @ W1 + b1
    prediction = h @ W2 + b2

    prediction_scalar = prediction.item()
    target_scalar = float(target)
    loss = (prediction_scalar - target_scalar) ** 2

    dloss_dprediction = 2 * (prediction_scalar - target_scalar)
    dprediction_dh = W2.reshape(-1)
    dloss_dh = dloss_dprediction * dprediction_dh

    return {
        "h": h,
        "prediction": prediction_scalar,
        "loss": loss,
        "dloss_dprediction": dloss_dprediction,
        "dprediction_dh": dprediction_dh,
        "dloss_dh": dloss_dh,
    }


trace = trace_single_request(
    x=X[0],
    target=y_true[0, 0],
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2,
)

print("=" * 70)
print("[필수 1-1: 첫 번째 표본 계산 그래프 추적]")
print("=" * 70)
print("h:", trace["h"])
print("prediction:", trace["prediction"])
print("loss:", trace["loss"])
print("dLoss/dPrediction:", trace["dloss_dprediction"])
print("dPrediction/dh:", trace["dprediction_dh"])
print("dLoss/dh:", trace["dloss_dh"])

print("\nnode | forward value | local gradient | upstream gradient | output gradient")
print(
    "MSE  | "
    f"prediction={trace['prediction']:.4f} | "
    f"2(pred-target)={trace['dloss_dprediction']:.4f} | "
    "1.0000 | "
    f"{trace['dloss_dprediction']:.4f}"
)
print(
    "W2   | "
    f"h={trace['h']} | "
    f"W2={trace['dprediction_dh']} | "
    f"{trace['dloss_dprediction']:.4f} | "
    f"{trace['dloss_dh']}"
)

# ----------------------------------------------------
# 필수 2. 두 단계 모델의 수동 Backpropagation (문제 2-1)
# ----------------------------------------------------
def forward_only(X, y_true, W1, b1, W2, b2):
    H = X @ W1 + b1
    predictions = H @ W2 + b2
    loss = float(np.mean((predictions - y_true) ** 2))
    return H, predictions, loss


def forward_backward(X, y_true, W1, b1, W2, b2):
    """두 단계 선형모델의 Forward와 Backward를 계산합니다."""
    H, predictions, loss = forward_only(X, y_true, W1, b1, W2, b2)

    n_samples = X.shape[0]
    d_predictions = 2 * (predictions - y_true) / n_samples

    dW2 = H.T @ d_predictions
    db2 = d_predictions.sum(axis=0)

    dH = d_predictions @ W2.T
    dW1 = X.T @ dH
    db1 = dH.sum(axis=0)

    return {
        "H": H,
        "predictions": predictions,
        "loss": loss,
        "d_predictions": d_predictions,
        "dH": dH,
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }


manual_result = forward_backward(X, y_true, W1, b1, W2, b2)

learning_rate = 0.2
W1_new = W1 - learning_rate * manual_result["dW1"]
b1_new = b1 - learning_rate * manual_result["db1"]
W2_new = W2 - learning_rate * manual_result["dW2"]
b2_new = b2 - learning_rate * manual_result["db2"]

_, predictions_after, loss_after = forward_only(
    X, y_true, W1_new, b1_new, W2_new, b2_new
)

print("\n" + "=" * 70)
print("[필수 2-1: 전체 배치 Backpropagation 및 파라미터 업데이트]")
print("=" * 70)
print("H shape:", manual_result["H"].shape)
print("prediction shape:", manual_result["predictions"].shape)
print("dPrediction shape:", manual_result["d_predictions"].shape)
print("dH shape:", manual_result["dH"].shape)
print("dW1 shape:", manual_result["dW1"].shape)
print("db1 shape:", manual_result["db1"].shape)
print("dW2 shape:", manual_result["dW2"].shape)
print("db2 shape:", manual_result["db2"].shape)

assert manual_result["dW1"].shape == W1.shape
assert manual_result["db1"].shape == b1.shape
assert manual_result["dW2"].shape == W2.shape
assert manual_result["db2"].shape == b2.shape
assert loss_after < manual_result["loss"]

print("\n[Backpropagation 검증 보고]")
print(f"- 업데이트 전 Loss: {manual_result['loss']:.6f}")
print(f"- 업데이트 후 Loss: {loss_after:.6f}")
print(f"- dW1 Shape: {manual_result['dW1'].shape}")
print(f"- db1 Shape: {manual_result['db1'].shape}")
print(f"- dW2 Shape: {manual_result['dW2'].shape}")
print(f"- db2 Shape: {manual_result['db2'].shape}")
print(f"- Loss 감소 여부: {loss_after < manual_result['loss']}")
print("- Backpropagation과 Gradient Descent의 역할 차이:")
print("  * Backpropagation: 연쇄 법칙(Chain Rule)을 이용하여 각 파라미터별 Gradient를 효율적으로 계산합니다.")
print("  * Gradient Descent: 계산된 Gradient 방향의 반대로 파라미터를 업데이트하여 Loss를 감소시킵니다.")
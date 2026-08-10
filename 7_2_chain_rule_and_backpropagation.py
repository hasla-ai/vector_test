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


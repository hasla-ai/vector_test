import numpy as np
from dataset_6_2 import Q, K, V, input_tokens


# ----------------------------------------------------
# 1. 세 버그 원인 설명
# ----------------------------------------------------
"""
[세 가지 버그 발생 원인 분석]

1. BUG 1 (Q.shape[0]으로 Scaling)
   - 원인: Key/Query의 벡터 차원인 d_k(Q.shape[-1]=3)가 아닌 시퀀스 길이 seq_len(Q.shape[0]=5)으로 나누었습니다.
   - 영향: 시퀀스 길이에 따라 Attention Score 스케일이 부적절하게 변하여, 차원 수 d_k에 따른 분산 안정화 효과를 얻지 못합니다.

2. BUG 2 (마스킹 반대 적용)
   - 원인: np.where(allowed_positions, -1e9, scores) 구문은 과거·현재(True)에 -1e9를 채우고 미래(False)에 원래 점수를 남깁니다.
   - 영향: 과거와 현재 정보는 차단되고, 미래 토큰만 참조하는 미래 토큰 누수(Information Leakage)가 발생합니다.

3. BUG 3 (axis=0 방향 Softmax)
   - 원인: Query별(행 방향, axis=-1) 확률 분포를 만들어야 하는데 열 방향(axis=0)으로 Softmax를 수행했습니다.
   - 영향: 각 Query(행)의 Weight 합이 1이 되지 않아 확률 해석이 불가능해집니다.
"""


# ----------------------------------------------------
# 2. 시작 코드 구현
# ----------------------------------------------------
def stable_softmax(values, axis=-1):
    """수치적으로 안정적인 Softmax 구현"""
    # 오버플로우 방지를 위한 max 차감 (keepdims=True로 차원 유지)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def fixed_causal_attention(Q, K, V):
    """Scaling → Mask → Softmax → @V 순서로 구현"""
    # 1. Scaling: key 차원 d_k(Q.shape[-1]) 사용
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)

    # 2. Masking: 하삼각(과거·현재)은 scores 유지, 상삼각(미래)은 -1e9 마스킹
    allowed_positions = np.tril(
        np.ones((Q.shape[0], K.shape[0]), dtype=bool)
    )
    masked_scores = np.where(allowed_positions, scores, -1e9)

    # 3. Softmax: 행 방향(axis=-1) 적용
    weights = stable_softmax(masked_scores, axis=-1)

    # 4. Context Matrix 계산
    context = weights @ V

    return scores, masked_scores, weights, context


def audit_attention_weights(weights):
    seq_len = weights.shape[0]
    future_indices = np.triu_indices(seq_len, k=1)

    return {
        "row_sums": weights.sum(axis=-1),
        "column_sums": weights.sum(axis=0),
        "future_max_weight": float(weights[future_indices].max()),
    }


# ----------------------------------------------------
# 3. 실행 및 진단 결과 출력
# ----------------------------------------------------
scores, masked_scores, weights, context = fixed_causal_attention(Q, K, V)
audit = audit_attention_weights(weights)

print("=" * 60)
print("[Causal Attention 진단 결과]")
print("=" * 60)
print("행별 합:", audit["row_sums"])
print("열별 합:", audit["column_sums"])
print("미래 위치 최대 Weight:", audit["future_max_weight"])
print("Context shape:", context.shape)

# 조건 만족 여부 검증
print("\n" + "=" * 60)
print("[검증 조건 확인]")
print("=" * 60)
print("1. 각 행의 합 == 1:", np.allclose(audit["row_sums"], 1.0))
print("2. 미래 위치 Weight == 0:", np.isclose(audit["future_max_weight"], 0.0))
print("3. Context Shape == (5, 2):", context.shape == (5, 2))

# 필수 2. 위치별 Cross Entropy로 실패 지점 찾기
## ▶ 문제 2-1: 다음 토큰 예측 품질 리포트 작성


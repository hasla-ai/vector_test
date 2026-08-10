import numpy as np
import pandas as pd
from dataset_6_2 import W_vocab, b_vocab, targets, input_tokens, vocabulary, Q, K, V


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
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def fixed_causal_attention(Q, K, V):
    """Causal Attention 연산 (Scaling -> Mask -> Softmax -> @V)"""
    """Scaling → Mask → Softmax → @V 순서로 구현"""
    # 1. Scaling: key 차원 d_k(Q.shape[-1]) 사용
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)

    # 2. Causal Masking: 하삼각(과거·현재)은 scores 유지, 상삼각(미래)은 -1e9 마스킹
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
    """Attention Weight 진단"""
    seq_len = weights.shape[0]
    future_indices = np.triu_indices(seq_len, k=1)

    return {
        "row_sums": weights.sum(axis=-1),
        "column_sums": weights.sum(axis=0),
        "future_max_weight": float(weights[future_indices].max()),
    }


# ----------------------------------------------------
# 3. 실행 및 진단 결과 출력(4개의 반환값을 모두 명시적으로 받기)
# ----------------------------------------------------
scores, masked_scores, weights, context = fixed_causal_attention(Q, K, V)

# Attention 진단 수행
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


def build_token_loss_report(
    context,
    W_vocab,
    b_vocab,
    targets,
    input_tokens,
    vocabulary,
):
    """위치별 다음 토큰 예측 결과와 NLL을 계산하고 리포트를 구성합니다."""
    # NumPy 배열 타입 및 차원 강제 보장 (에러 방지 방어 코드)
    context = np.asarray(context, dtype=np.float64)
    W_vocab = np.asarray(W_vocab, dtype=np.float64)
    b_vocab = np.asarray(b_vocab, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    # 1. Logits 계산: (seq_len, vocab_size) (5, 2) @ (2, 7) + (7,) -> (5, 7)
    logits = context @ W_vocab + b_vocab

    # 2. Vocabulary 차원에 Softmax 적용하여 확률 분포 산출
    probs = stable_softmax(logits, axis=-1)

    report_rows = []
    nll_list = []

    for i in range(len(input_tokens)):
        in_tok = input_tokens[i]
        target_idx = targets[i]
        exp_tok = vocabulary[target_idx]

        # 정답 토큰 확률 선택
        target_prob = probs[i, target_idx]
        
        # NLL = -log(정답 확률)
        nll = -np.log(max(target_prob, 1e-15))
        nll_list.append(nll)

        # 모델이 가장 높게 예측한 토큰 (Top-1)
        pred_idx = np.argmax(probs[i])
        pred_tok = vocabulary[pred_idx]

        report_rows.append({
            "position": i,
            "input_token": in_tok,
            "expected_next_token": exp_tok,
            "predicted_token": pred_tok,
            "target_probability": float(target_prob),
            "NLL": float(nll)
        })

    df_report = pd.DataFrame(report_rows)
    mean_ce = float(np.mean(nll_list))
    max_nll_idx = int(np.argmax(nll_list))

    return {
        "df_report": df_report,
        "mean_ce": mean_ce,
        "max_nll_idx": max_nll_idx
    }


loss_report = build_token_loss_report(
    context=context,
    W_vocab=W_vocab,
    b_vocab=b_vocab,
    targets=targets,
    input_tokens=input_tokens,
    vocabulary=vocabulary,
)


print("=" * 70)
print("[문제 2-1: 위치별 토큰 예측 결과 표]")
print("=" * 70)
print(loss_report["df_report"].to_string(index=False))
print(f"\n평균 Cross Entropy (Mean CE): {loss_report['mean_ce']:.4f}")

# 가장 NLL이 큰 위치 정보 추출
max_idx = loss_report["max_nll_idx"]
target_row = loss_report["df_report"].iloc[max_idx]

# ----------------------------------------------------
# [다음 토큰 예측 품질 보고] 작성
# ----------------------------------------------------
print("\n" + "=" * 70)
print("[다음 토큰 예측 품질 보고]")
print("=" * 70)
print(f"- 평균 Cross Entropy: {loss_report['mean_ce']:.4f}")
print(f"- 가장 큰 NLL 위치: index {max_idx}")
print(f"- 입력 토큰: {target_row['input_token']}")
print(f"- 기대한 다음 토큰: {target_row['expected_next_token']}")
print(f"- 실제 Top-1 예측: {target_row['predicted_token']}")
print(f"- 정답 확률: {target_row['target_probability']:.4f}")
print(f"- 개선이 필요한 이유: 모델이 '재설정' 입력 시점에 정답인 '방법' 대신 오답인 '사내'를 가장 높은 확률로 예측하여 NLL이 2.0039로 가장 높게 치솟았습니다. 현재 초기화된 임베딩과 가중치(W_vocab) 상에서 특정 토큰 편향이 존재하므로, 언어 모델 가중치 학습(Training) 과정에서 출력 프로젝션 파라미터의 정규화 및 크로스 엔트로피 손실 기반 최적화가 필수적입니다.")
print("=" * 70)


# 심화 1. Temperature 정책 비교하기
## ▶ 문제 3-1: 내부 업무 챗봇의 확률분포 변화 분석

def compare_temperature(logits, target_index, temperatures, vocabulary):
    """Temperature별 확률분포 요약을 반환하세요."""
    logits = np.asarray(logits, dtype=np.float64)
    results = []

    for T in temperatures:
        # 1. Temperature 적용 (Logits / T)
        scaled_logits = logits / T

        # 2. Softmax 적용하여 확률분포 산출
        probs = stable_softmax(scaled_logits, axis=-1)

        # 3. Top-1 토큰 및 확률
        top1_idx = np.argmax(probs)
        top1_token = vocabulary[top1_idx]
        top1_prob = float(probs[top1_idx])

        # 4. 정답 토큰 확률
        target_prob = float(probs[target_index])

        # 5. 엔트로피(Entropy) 계산: -sum(p * log(p))
        # 수치 안정성을 위해 p=0 인 경우 log(0)=NaN 방지 처리 (1e-15 적용)
        safe_probs = np.clip(probs, 1e-15, 1.0)
        entropy = float(-np.sum(probs * np.log(safe_probs)))

        results.append({
            "temperature": T,
            "top1_token": top1_token,
            "top1_prob": top1_prob,
            "target_prob": target_prob,
            "entropy": entropy,
        })

    return pd.DataFrame(results)

# ----------------------------------------------------
# 심화 3-1 실행 및 비교 분석
# ----------------------------------------------------
# 1. 문제 2-1에서 구한 가장 큰 NLL 위치(index 3, '재설정')의 Logits 가져오기
logits_all = context @ W_vocab + b_vocab
max_nll_idx = loss_report["max_nll_idx"]  # index 3
target_logits = logits_all[max_nll_idx]
target_idx = targets[max_nll_idx]  # 정답 index 4 ('방법')

temperatures = [0.6, 1.0, 1.4]

df_temp_summary = compare_temperature(
    logits=target_logits,
    target_index=target_idx,
    temperatures=temperatures,
    vocabulary=vocabulary,
)

print("=" * 70)
print("[Temperature별 확률분포 요약 표]")
print("=" * 70)
print(df_temp_summary.to_string(index=False))
print("=" * 70)
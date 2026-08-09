import numpy as np
import pandas as pd
import platform
import matplotlib.pyplot as plt
from dataset_6_1 import X_batch, X, W_Q, W_K, tokens, batch_tokens

# 필수 1. “재설정” 토큰이 무엇을 참고하는지 진단하기
## 문제 1-1: Attention Score 진단 함수 완성

def build_attention_diagnostics(X, W_Q, W_K, tokens, query_index):
    """Attention Score를 계산하고 진단에 필요한 결과를 딕셔너리로 반환합니다."""
    # 1. Q, K 계산
    Q = X @ W_Q
    K = X @ W_K

    # 2. raw_scores 계산
    raw_scores = Q @ K.T

    # 3. d_k 기반 scaling 계산
    d_k = Q.shape[-1]
    scaled_scores = raw_scores / np.sqrt(d_k)

    # 4. query_index 행 선택
    query_scores = scaled_scores[query_index]

    # 5. 자기 자신 위치 제외 가장 큰 Key index 탐색
    masked_scores = query_scores.copy()
    masked_scores[query_index] = -np.inf
    top_key_index = int(np.argmax(masked_scores))

    # 6. 토큰별 순위 매기기 (내림차순 정렬)
    raw_query_scores = raw_scores[query_index]
    sorted_indices = np.argsort(-query_scores)

    ranking = []
    for rank, idx in enumerate(sorted_indices, start=1):
        ranking.append({
            "rank": rank,
            "key_index": idx,
            "key_token": tokens[idx],
            "raw_score": raw_query_scores[idx],
            "scaled_score": query_scores[idx],
        })

    return {
        "Q": Q,
        "K": K,
        "raw_scores": raw_scores,
        "scaled_scores": scaled_scores,
        "query_scores": query_scores,
        "ranking": ranking,
        "top_key_index": top_key_index,
    }


# 실행 및 출력
report = build_attention_diagnostics(
    X=X,
    W_Q=W_Q,
    W_K=W_K,
    tokens=tokens,
    query_index=3,
)

# 1. Shape 검증 및 기본 출력
print("Q shape:", report["Q"].shape)
print("K shape:", report["K"].shape)
print("Score shape:", report["scaled_scores"].shape)
print("자기 자신 제외 최상위 Key:", tokens[report["top_key_index"]])

# 2. 조건 만족 검증
print("\n[조건 검증]")
print("1) Q.shape == K.shape == (5, 3) :", report["Q"].shape == (5, 3) and report["K"].shape == (5, 3))
print("2) raw_scores.shape == scaled_scores.shape == (5, 5) :", report["raw_scores"].shape == (5, 5) and report["scaled_scores"].shape == (5, 5))

raw_std = np.std(report["raw_scores"])
scaled_std = np.std(report["scaled_scores"])
print(f"3) Scaling 후 표준편차가 더 작음 : {scaled_std < raw_std} (전: {raw_std:.4f} -> 후: {scaled_std:.4f})")

# 3. 제출 결과 표 출력
print("\n[Attention Ranking 표]")
df_ranking = pd.DataFrame(report["ranking"])
print(df_ranking.to_string(index=False))

# 4. Attention 진단 보고서 작성
print("\n" + "=" * 50)
print("[Attention 진단 보고]")
print(f"- Query 토큰: {tokens[3]}")
print(f"- 자기 자신 제외 최상위 Key: {tokens[report['top_key_index']]}")
print(f"- Scaling 전 Score 표준편차: {raw_std:.4f}")
print(f"- Scaling 후 Score 표준편차: {scaled_std:.4f}")
print(f"- 품질 담당자에게 전달할 해석: '재설정' 토큰은 자기 자신을 제외했을 때 '비밀번호' 토큰과의 연관도 점수(Scaled Score: {report['scaled_scores'][3, 2]:.4f})가 가장 높습니다. 따라서 모델이 재설정의 대상이 되는 핵심 객체인 '비밀번호'를 정확하게 문맥상 연결하여 참고하고 있음을 확인할 수 있습니다.")
print("=" * 50)

# 필수 2. 배치 입력에서 잘못된 전치 연산 수정하기
## 문제 2-1: K.T 때문에 발생한 Shape 오류 디버깅

import numpy as np
import pandas as pd
from dataset_6_1 import X_batch, W_Q, W_K, batch_tokens


def buggy_batch_attention_scores(X_batch, W_Q, W_K):
    """버그가 발생하는 배치 Attention Score 함수입니다."""
    Q_batch = X_batch @ W_Q  # (2, 5, 3)
    K_batch = X_batch @ W_K  # (2, 5, 3)

    # 버그 발생 원인: 3차원 텐서에서 .T는 모든 축의 순서를 뒤집어 (0, 1, 2) -> (2, 1, 0)으로 바꿉니다.
    # 따라서 (2, 5, 3)의 전치 결과는 (3, 5, 2)가 되며, (2, 5, 3) @ (3, 5, 2) 행렬곱 시
    # 내적 축(inner dimension)의 크기(3과 5)가 일치하지 않아 ValueError가 발생합니다.
    return Q_batch @ K_batch.T


def build_batch_attention_scores(X_batch, W_Q, W_K):
    """np.swapaxes를 적용하여 올바르게 수정한 배치 Attention Score 함수입니다."""
    # 1. Q_batch, K_batch 계산: (2, 5, 3)
    Q_batch = X_batch @ W_Q
    K_batch = X_batch @ W_K

    # 2. 마지막 두 축만 전치: (2, 5, 3) -> (2, 3, 5)
    K_batch_T = np.swapaxes(K_batch, -1, -2)

    # 3. raw_scores_batch 계산: (2, 5, 3) @ (2, 3, 5) -> (2, 5, 5)
    raw_scores_batch = Q_batch @ K_batch_T

    # 4. Scaling 적용
    d_k = Q_batch.shape[-1]
    scaled_scores_batch = raw_scores_batch / np.sqrt(d_k)

    return scaled_scores_batch


# 1. 버그 원인 설명 및 예외 확인
print("=" * 60)
print("[1. buggy_batch_attention_scores 원인 분석]")
print("=" * 60)
try:
    buggy_batch_attention_scores(X_batch, W_Q, W_K)
except ValueError as e:
    print(f"버그 함수 실행 예외 발생: {e}")
    print("원인: K_batch(2, 5, 3)에 .T 적용 시 (3, 5, 2)가 되어 (2, 5, 3) @ (3, 5, 2) 내적 차원(3 vs 5)이 불일치합니다.")

# 2. 수정된 함수 실행 및 Shape 검증
print("\n" + "=" * 60)
print("[2. build_batch_attention_scores 검증]")
print("=" * 60)
scaled_scores_batch = build_batch_attention_scores(X_batch, W_Q, W_K)
print("scaled_scores_batch shape:", scaled_scores_batch.shape)
print("scaled_scores_batch.shape == (2, 5, 5) :", scaled_scores_batch.shape == (2, 5, 5))

# 3. 배치별 주요 Query 토큰 진단
# 첫 번째 배치(0) : "재설정" (index 3)
# 두 번째 배치(1) : "요청" (index 4)
targets = [
    (0, 3, "재설정"),
    (1, 4, "요청")
]

report_rows = []

for b_idx, q_idx, q_token in targets:
    scores = scaled_scores_batch[b_idx, q_idx].copy()
    
    # 자기 자신 제외
    scores[q_idx] = -np.inf
    top_key_idx = int(np.argmax(scores))
    top_key_token = batch_tokens[b_idx][top_key_idx]
    top_score = scaled_scores_batch[b_idx, q_idx, top_key_idx]

    report_rows.append({
        "batch": b_idx,
        "query_token": q_token,
        "top_nonself_key": top_key_token,
        "score": round(top_score, 4)
    })

# 4. 배치 진단 보고 출력
print("\n" + "=" * 60)
print("[배치 진단 보고]")
print("=" * 60)
df_report = pd.DataFrame(report_rows)
print(df_report.to_string(index=False))

# 심화1. d_k가 커질 때 Scaling의 효과를 수치로 확인하기
## 문제 3-1: Score 분산 모니터링 실험

# 한글 폰트 설정 (환경에 맞춰 선택)
plt.rc('font', family='Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic')

plt.rcParams['axes.unicode_minus'] = False


def run_scaling_experiment(d_k_values, trials=5000, seed=42):
    """차원별 raw/scaled score 표준편차를 계산하여 반환합니다."""
    np.random.seed(seed)
    results = []

    for d_k in d_k_values:
        # 1. 표준정규분포에서 Query, Key 생성 (trials, d_k)
        Q = np.random.randn(trials, d_k)
        K = np.random.randn(trials, d_k)

        # 2. 각 쌍의 내적 계산 (행별 내적: sum(Q * K, axis=1))
        raw_scores = np.sum(Q * K, axis=1)

        # 3. Scaling 적용
        scaled_scores = raw_scores / np.sqrt(d_k)

        # 4. 표준편차 기록
        results.append({
            "d_k": d_k,
            "raw_std": np.std(raw_scores),
            "scaled_std": np.std(scaled_scores)
        })

    return pd.DataFrame(results)


# 실험 실행
d_k_values = [4, 16, 64, 256]
df_results = run_scaling_experiment(d_k_values)

# 결과 수치 출력
print("=" * 60)
print("[차원(d_k)별 Score 표준편차 모니터링 결과]")
print("=" * 60)
print(df_results.to_string(index=False))

# ----------------------------------------------------
# 시각화 (막대그래프)
# ----------------------------------------------------
x = np.arange(len(d_k_values))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, df_results["raw_std"], width, label='Scaling 전 (Raw Std)')
rects2 = ax.bar(x + width/2, df_results["scaled_std"], width, label='Scaling 후 (Scaled Std)')

ax.set_xlabel('d_k (Key Dimension)')
ax.set_ylabel('Standard Deviation (표준편차)')
ax.set_title('d_k 차원 변화에 따른 Attention Score 분산 안정화 비교')
ax.set_xticks(x)
ax.set_xticklabels(d_k_values)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
from dataset_6_1 import X, W_Q, W_K, tokens

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
6장 1강 실습  Attention Score와 QKT Shape

# 필수 1. “재설정” 토큰이 무엇을 참고하는지 진단하기

## ▶ 문제 1-1: Attention Score 진단 함수 완성

### 업무 요청

품질 담당자는 “재설정” 토큰이 자기 자신을 제외했을 때 **“비밀번호” 토큰을 가장 많이 참고하는지** 확인하려고 합니다. 단순히 점수만 출력하지 말고, 토큰별 관련도 순위와 Scaling 전후 점수를 함께 정리하세요.

### 수행해야 할 작업

`build_attention_diagnostics()` 함수의 전체 계산 흐름을 완성하세요.

1. `Q = X @ W_Q`, `K = X @ W_K`를 계산하세요.
2. `raw_scores = Q @ K.T`를 계산하세요.
3. `d_k = Q.shape[-1]`로 구한 뒤 `scaled_scores = raw_scores / np.sqrt(d_k)`를 계산하세요.
4. `query_index=3`인 “재설정” 행을 선택하세요.
5. 자기 자신 위치의 점수만 제외하고 가장 큰 Key index를 찾으세요.
6. 토큰별 `raw_score`, `scaled_score`, 순위를 출력하세요.
7. 다음 조건을 만족하는지 확인하세요.
    - `Q.shape == K.shape == (5, 3)`
    - `raw_scores.shape == scaled_scores.shape == (5, 5)`
    - Scaling 후 표준편차가 더 작음

### 시작 코드

```python
def build_attention_diagnostics(X, W_Q, W_K, tokens, query_index):
    """
    Attention Score를 계산하고 진단에 필요한 결과를 딕셔너리로 반환하세요.

    반환할 key:
    - Q, K
    - raw_scores, scaled_scores
    - query_scores
    - ranking
    - top_key_index
    """
    #TODO: 이 함수의 전체 계산 흐름을 완성하세요.
    raise NotImplementedError

report = build_attention_diagnostics(
    X=X,
    W_Q=W_Q,
    W_K=W_K,
    tokens=tokens,
    query_index=3,
)

print("Q shape:", report["Q"].shape)
print("K shape:", report["K"].shape)
print("Score shape:", report["scaled_scores"].shape)
print("자기 자신 제외 최상위 Key:", tokens[report["top_key_index"]])
```

### 제출해야 할 결과

아래 형태의 표를 출력하세요.

```
rank | key_index | key_token | raw_score | scaled_score
```

그리고 다음 보고 문장을 실제 결과에 맞게 작성하세요.

```
[Attention 진단 보고]
- Query 토큰:
- 자기 자신 제외 최상위 Key:
- Scaling 전 Score 표준편차:
- Scaling 후 Score 표준편차:
- 품질 담당자에게 전달할 해석:
```

- `Q @ K.T`의 행은 Query 위치, 열은 Key 위치입니다.
- 자기 자신을 순위에서 제외할 때 원본 배열을 직접 바꾸지 말고 `copy()`를 사용하세요.
- 제외할 위치에는 `np.inf`를 넣으면 `argmax`에서 선택되지 않습니다.
- 내림차순 정렬은 `np.argsort(scores)[::-1]`로 만들 수 있습니다.

수행 결과

```bash
X shape: (5, 4)
W_Q shape: (4, 3)
W_K shape: (4, 3)
W_V shape: (4, 2)
Q shape: (5, 3)
K shape: (5, 3)
Score shape: (5, 5)
자기 자신 제외 최상위 Key: 비밀번호

[조건 검증]
1) Q.shape == K.shape == (5, 3) : True
2) raw_scores.shape == scaled_scores.shape == (5, 5) : True
3) Scaling 후 표준편차가 더 작음 : True (전: 0.3773 -> 후: 0.2178)

[Attention Ranking 표]
 rank  key_index key_token  raw_score  scaled_score
    1          3       재설정     1.6088      0.928841
    2          2      비밀번호     1.0904      0.629543
    3          4        방법     0.9328      0.538552
    4          1        사내     0.5732      0.330937
    5          0     <BOS>     0.2588      0.149418

==================================================
[Attention 진단 보고]
- Query 토큰: 재설정
- 자기 자신 제외 최상위 Key: 비밀번호
- Scaling 전 Score 표준편차: 0.3773
- Scaling 후 Score 표준편차: 0.2178
- 품질 담당자에게 전달할 해석: '재설정' 토큰은 자기 자신을 제외했을 때 '비밀번호' 토큰과의 연관도 점수(Scaled Score: 0.6295)가 가장 높습니다. 따라서 모델이 재설정의 대상이 되는 핵심 객체인 '비밀번호'를 정확하게 문맥상 연결하여 참고하고 있음을 확인할 수 있습니다.
==================================================
```

√d_k로 나누어도 한 행 안의 순위는 바뀌지 않습니다. 모든 점수에 같은 양의 상수를 나누기 때문입니다. 다만 점수의 전체 크기와 분산이 줄어들어 이후 Softmax가 지나치게 뾰족해지는 위험을 낮춥니다.

# 필수 2. 배치 입력에서 잘못된 전치 연산 수정하기

## ▶ 문제 2-1: `K.T` 때문에 발생한 Shape 오류 디버깅

### 업무 상황

운영 환경에서는 여러 문의를 한 번에 묶어 처리합니다. 동료가 작성한 아래 함수는 단일 시퀀스에서는 익숙해 보이지만, 3차원 배치 입력에서 실행하면 Shape 오류가 발생합니다.

```python
def buggy_batch_attention_scores(X_batch, W_Q, W_K):
    Q_batch = X_batch @ W_Q
    K_batch = X_batch @ W_K

    # 버그: 3차원 텐서에서 .T는 마지막 두 축만 바꾸지 않습니다.
    return Q_batch @ K_batch.T
```

# 필수 2. 배치 입력에서 잘못된 전치 연산 수정하기

## ▶ 문제 2-1: `K.T` 때문에 발생한 Shape 오류 디버깅

### 업무 상황

운영 환경에서는 여러 문의를 한 번에 묶어 처리합니다. 동료가 작성한 아래 함수는 단일 시퀀스에서는 익숙해 보이지만, 3차원 배치 입력에서 실행하면 Shape 오류가 발생합니다.

```python
def buggy_batch_attention_scores(X_batch, W_Q, W_K):
    Q_batch = X_batch @ W_Q
    K_batch = X_batch @ W_K

    # 버그: 3차원 텐서에서 .T는 마지막 두 축만 바꾸지 않습니다.
    return Q_batch @ K_batch.T
```

### 제공 데이터

```python
batch_tokens = [
    ["<BOS>", "사내", "비밀번호", "재설정", "방법"],
    ["<BOS>", "GCP", "GPU", "할당", "요청"],
]

X_2 = np.array([
    [0.1, 0.1, 0.0, 0.4],
    [0.3, 0.9, 0.1, 0.3],
    [0.4, 0.8, 0.4, 0.2],
    [0.3, 0.6, 0.9, 0.3],
    [0.5, 0.5, 0.8, 0.7],
], dtype=float)

X_batch = np.stack([X, X_2], axis=0)
print("X_batch shape:", X_batch.shape)
```

### 수행해야 할 작업

1. `buggy_batch_attention_scores()`가 왜 잘못되었는지 Shape으로 설명하세요.
2. 마지막 두 축만 바꾸는 `np.swapaxes(K_batch, -1, -2)`를 사용해 함수를 수정하세요.
3. `scaled_scores_batch.shape == (2, 5, 5)`인지 확인하세요.
4. 첫 번째 문의의 “재설정”, 두 번째 문의의 “요청” Query가 자기 자신을 제외하고 가장 크게 참고한 토큰을 출력하세요.
5. 다음 형태의 배치 진단 보고를 작성하세요.

```
batch | query_token | top_nonself_key | score
```
Q_batch.shape   = (batch, seq_len, d_k)
K_batch.shape   = (batch, seq_len, d_k)
K_batch_T.shape = (batch, d_k, seq_len)

- 2차원 행렬의 `.T`는 행과 열을 바꾸지만, 3차원 배열의 `.T`는 모든 축의 순서를 뒤집습니다.
- 배치 축은 그대로 두고 마지막 두 축만 바꾸세요.

```bash
============================================================
[1. buggy_batch_attention_scores 원인 분석]
============================================================
버그 함수 실행 예외 발생: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 5 is different from 3)
원인: K_batch(2, 5, 3)에 .T 적용 시 (3, 5, 2)가 되어 (2, 5, 3) @ (3, 5, 2) 내적 차원(3 vs 5)이 불일치합니다.

============================================================
[2. build_batch_attention_scores 검증]
============================================================
scaled_scores_batch shape: (2, 5, 5)
scaled_scores_batch.shape == (2, 5, 5) : True

============================================================
[배치 진단 보고]
============================================================
 batch query_token top_nonself_key  score
     0         재설정            비밀번호 0.6295
     1          요청              할당 0.7831
```

핵심 Shape은 다음과 같습니다.

```
Q_batch shape: (2, 5, 3)
K_batch shape: (2, 5, 3)
K_batch_T shape: (2, 3, 5)
scaled_scores_batch shape: (2, 5, 5)
```

첫 번째 문의에서는 “재설정”이 자기 자신을 제외하고 “비밀번호”를, 두 번째 문의에서는 “요청”이 “할당”을 가장 크게 참고합니다.
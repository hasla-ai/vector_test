[6장 2강] - 실습: Softmax와 Cross Entropy

필수 1. 미래 토큰 누수 버그 수정하기

▶ 문제 1-1: 세 가지 구현 오류를 찾아 Causal Attention 복구

장애 티켓

아래 함수는 실행되지만 결과가 이상합니다.

Attention Weight의 행별 합이 1이 아닙니다.
현재 위치가 미래 토큰을 참고합니다.
Scaling에 사용한 차원이 d_k와 맞지 않습니다.

```bash 
def buggy_causal_attention(Q, K, V):
    #BUG 1: seq_len을 사용해 Scaling하고 있습니다.
    scores = (Q @ K.T) / np.sqrt(Q.shape[0])

    #BUG 2: 허용해야 할 과거·현재 위치를 가리고 있습니다.
    allowed_positions = np.tril(
        np.ones((Q.shape[0], K.shape[0]), dtype=bool)
    )
    masked_scores = np.where(allowed_positions, -1e9, scores)

    #BUG 3: Query별 확률이 아니라 열 방향으로 Softmax를 적용합니다.
    shifted = masked_scores - np.max(masked_scores, axis=0, keepdims=True)
    exp_values = np.exp(shifted)
    weights = exp_values / np.sum(exp_values, axis=0, keepdims=True)

    context = weights @ V
    return scores, masked_scores, weights, context

```

수행해야 할 작업

세 버그가 각각 어떤 문제를 일으키는지 설명하세요.
stable_softmax(values, axis=-1) 함수를 작성하세요.
fixed_causal_attention()을 올바른 순서로 구현하세요.
다음 진단값을 출력하세요.
    행별 Weight 합
    미래 위치의 최대 Weight
    Context Shape

다음 조건을 모두 만족해야 합니다.
    각 행의 합 = 1
    미래 위치 Weight = 0에 가까움
    Context Shape = (5, 2)
    ​
진단 도우미 코드

```bash
def audit_attention_weights(weights):
    seq_len = weights.shape[0]
    future_indices = np.triu_indices(seq_len, k=1)

    return {
        "row_sums": weights.sum(axis=-1),
        "column_sums": weights.sum(axis=0),
        "future_max_weight": float(weights[future_indices].max()),
    }

```

시작 코드

```bash
def stable_softmax(values, axis=-1):
    """수치적으로 안정적인 Softmax를 구현하세요."""
    #TODO
    raise NotImplementedError


def fixed_causal_attention(Q, K, V):
    """Scaling → Mask → Softmax → @V 순서로 구현하세요."""
    #TODO
    raise NotImplementedError


scores, masked_scores, weights, context = fixed_causal_attention(Q, K, V)
audit = audit_attention_weights(weights)

print("행별 합:", audit["row_sums"])
print("열별 합:", audit["column_sums"])
print("미래 위치 최대 Weight:", audit["future_max_weight"])
print("Context shape:", context.shape)

```

- `d_k`는 시퀀스 길이가 아니라 `Q.shape[-1]`입니다.
- 미래 위치는 행 `i`에서 열 `j > i`인 상삼각 영역입니다.
- `np.triu(..., k=1)`은 대각선 위쪽만 `True`로 만듭니다.
- Attention에서는 한 Query가 모든 Key에 배분하는 확률이므로 마지막 축의 합이 1이어야 합니다.

결과

```bash
============================================================
[Causal Attention 진단 결과]
============================================================
행별 합: [1. 1. 1. 1. 1.]
열별 합: [2.0768 1.2146 0.8798 0.6156 0.2133]
미래 위치 최대 Weight: 0.0
Context shape: (5, 2)

============================================================
[검증 조건 확인]
============================================================
1. 각 행의 합 == 1: True
2. 미래 위치 Weight == 0: True
3. Context Shape == (5, 2): True
```

버그 함수는 axis=0으로 Softmax를 적용하므로 열별 합이 1이 되고, 행별 합은 [2.1620, 1.2789, 0.8555, 0.5037, 0.2000]처럼 무너집니다. 또한 허용 위치를 가렸기 때문에 미래 위치에 큰 Weight가 남습니다.

# 필수 2. 위치별 Cross Entropy로 실패 지점 찾기

## ▶ 문제 2-1: 다음 토큰 예측 품질 리포트 작성

### 업무 요청

Causal Attention이 정상화되었습니다. 이제 각 위치의 Context를 Vocabulary Logits로 바꾸고, **어느 위치에서 정답 토큰 확률이 가장 낮은지** 확인해야 합니다.

평균 Cross Entropy 하나만 보고하면 원인을 찾기 어렵습니다. 위치별로 다음 정보를 정리하세요.

```bash
position
input_token
expected_next_token
predicted_token
target_probability
NLL
```

시작코드

```bash
def build_token_loss_report(
    context,
    W_vocab,
    b_vocab,
    targets,
    input_tokens,
    vocabulary,
):
    """위치별 다음 토큰 예측 결과와 NLL을 반환하세요."""
    #TODO: 계산 흐름 전체를 구현하세요.
    raise NotImplementedError


loss_report = build_token_loss_report(
    context=context,
    W_vocab=W_vocab,
    b_vocab=b_vocab,
    targets=targets,
    input_tokens=input_tokens,
    vocabulary=vocabulary,
)

```

### 수행해야 할 작업

1. `logits = context @ W_vocab + b_vocab`를 계산하세요.
2. 마지막 축에 Vocabulary Softmax를 적용하세요.
3. 각 위치의 정답 토큰 확률을 선택하세요.
4. `NLL = -log(정답 확률)`을 계산하세요.
5. 평균 Cross Entropy를 계산하세요.
6. NLL이 가장 큰 위치를 찾으세요.
7. 해당 위치에서 모델이 가장 높게 예측한 토큰과 그 확률을 출력하세요.
8. 결과를 바탕으로 3~4문장의 품질 점검 보고서를 작성하세요.

### 제출해야 할 보고 형식

```
[다음 토큰 예측 품질 보고]
- 평균 Cross Entropy:
- 가장 큰 NLL 위치:
- 입력 토큰:
- 기대한 다음 토큰:
- 실제 Top-1 예측:
- 정답 확률:
- 개선이 필요한 이유:
```
- 정답 확률 선택: `probs[np.arange(len(targets)), targets]`
- 예측 index: `np.argmax(probs, axis=-1)`
- 가장 큰 손실 위치: `np.argmax(position_nll)`
- 로그 0을 피하려면 `np.clip(probability, 1e-12, 1.0)`을 사용할 수 있습니다.

```결과
======================================================================
[문제 2-1: 위치별 토큰 예측 결과 표]
======================================================================
 position input_token expected_next_token predicted_token  target_probability      NLL
        0       <BOS>                  사내              사내            0.161611 1.822564
        1          사내                비밀번호              사내            0.139213 1.971749
        2        비밀번호                 재설정              사내            0.145139 1.930061
        3         재설정                  방법              사내            0.134796 2.003994
        4          방법               <EOS>              사내            0.156383 1.855448

평균 Cross Entropy (Mean CE): 1.9168

======================================================================
[다음 토큰 예측 품질 보고]
======================================================================
- 평균 Cross Entropy: 1.9168
- 가장 큰 NLL 위치: index 3
- 입력 토큰: 재설정
- 기대한 다음 토큰: 방법
- 실제 Top-1 예측: 사내
- 정답 확률: 0.1348
- 개선이 필요한 이유: 모델이 '재설정' 입력 시점에 정답인 '방법' 대신 오답인 '사내'를 가장 높은 확률로 예측하여 NLL이 2.0039로 가장 높게 치솟았습니다. 현재 초기화된 임베딩과 가중치(W_vocab) 상에서 특정 토큰 편향이 존재하므로, 언어 모델 가중치 학습(Training) 과정에서 출력 프로젝션 파라미터의 정규화 및 크로스 엔트로피 손실 기반 최적화가 필수적입니다.
======================================================================
```

NLL이 크다는 것은 정답 토큰에 낮은 확률을 배정했다는 뜻입니다. 평균 Loss만 보면 위치별 문제를 놓칠 수 있으므로, 실제 운영에서는 샘플·위치·토큰별 Loss와 오류 유형을 함께 확인합니다.

# 심화 1. Temperature 정책 비교하기

## ▶ 문제 3-1: 내부 업무 챗봇의 확률분포 변화 분석

### 업무 상황

운영팀은 답변이 너무 획일적이면 유연성이 떨어지고, 너무 무작위적이면 업무 안내의 신뢰성이 낮아질 수 있다고 우려합니다. 가장 큰 NLL이 발생한 위치의 Logits에 서로 다른 Temperature를 적용해 분포를 비교하세요.

### 수행해야 할 작업

1. Temperature `0.6`, `1.0`, `1.4`를 비교하세요.
2. 각 Temperature에서 다음을 계산하세요.
    - Top-1 토큰
    - Top-1 확률
    - 정답 토큰 확률
    - Entropy
3. Temperature가 낮아질수록 분포가 어떻게 변하는지 설명하세요.
4. 사내 절차 안내 챗봇에 사용할 값을 하나 추천하고, 이유를 작성하세요.

시작코드 

```bash
def compare_temperature(logits, target_index, temperatures, vocabulary):
    """Temperature별 확률분포 요약을 반환하세요."""
    #TODO
    raise NotImplementedError
```

결과

```bash

```


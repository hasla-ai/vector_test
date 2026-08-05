# UCI Wine Quality 데이터셋 분석 및 파이썬 벡터 실습

제조 품질 데이터(UCI Wine Quality) 분석을 위한 표준 Python 프로젝트 개발 환경입니다.  
`uv`를 활용하여 의존성 관리를 표준화하고 이화학 데이터 기반의 벡터 노름 및 정규화 연산을 수행합니다.

## 🛠️ 개발 환경 및 기술 스택
- **Language**: Python >= 3.10
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Data & ML Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `ucimlrepo`

## 📂 프로젝트 구조
```text
vector_test/
├── .gitignore          # Git 제외 대상 설정 (.venv 등)
├── pyproject.toml      # 프로젝트 정보 및 핵심 의존성 명세
├── uv.lock             # 의존성 패키지 버전 잠금 파일
├── README.md           # 프로젝트 문서
└── vector_test.py      # 벡터 노름 및 정규화 검증 스크립트

🚀 프로젝트 실행 방법
1. Repository Clone

```bash
git clone [https://github.com/hasla-ai/vector_test.git](https://github.com/hasla-ai/vector_test.git)
cd vector_test
```

2. uv를 활용한 가상환경 및 의존성 동기화
uv.lock 파일 기반으로 100% 동일한 개발 환경을 구축합니다.

```bash
uv sync
```

3. 스크립트 실행

```bash
uv run vector_test.py
```

---

## 2. 다음 단계 실습 방향 (제조 품질 데이터 EDA 및 모델링)

환경 구축과 기본 벡터 연산 실습이 완료되었으므로, 본격적인 제조 품질 데이터 분석 단계로 나아갈 수 있습니다.

### 추천 다음 단계

1. **이화학 데이터 탐색적 데이터 분석 (EDA)**
   * 와인의 알코올 도수, 산도(pH), 잔당(residual sugar) 등 수치형 변수 간 상관관계 분석 및 `matplotlib`/`seaborn` 시각화

2. **품질(Quality) 분류/회귀 모델 구축**
   * `scikit-learn`을 활용하여 이화학 측정값을 기반으로 와인 품질 등급을 예측하는 ML 모델 학습 (Random Forest, XGBoost 등)

3. **특성 스케일링(Scaling) 비교**
   * 앞서 구현한 Vector Norm 개념을 바탕으로, `StandardScaler`와 `MinMaxScaler` 적용 전후의 ML 모델 성능 변화 비교
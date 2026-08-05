import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimprepo import fetch_ucirepo

# ---------------------------------------------------------
# 1. 데이터셋 로드 (UCI Wine Quality ID: 186)
# ---------------------------------------------------------
wine_quality = fetch_ucirepo(id=186)

# 특성 데이터(X) 및 타깃 데이터(y) 추출
X = wine_quality.data.features
y = wine_quality.data.targets

print("=== 데이터셋 기본 정보 ===")
print(f"특성 데이터 행렬 크기: {X.shape}")
print("특성 이름:", list(X.columns))
print("\n[상위 5개 샘플 데이터]")
print(X.head())
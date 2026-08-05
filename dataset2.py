import numpy as np
import pandas as pd

# 공통 난수 시드 설정
RANDOM_STATE = 42


def _dummy_retail():
    """UCI를 쓸 수 없을 때 사용하는 동일한 컬럼 구조의 대체 거래 데이터입니다."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 6000
    return pd.DataFrame({
        'InvoiceNo': rng.integers(10000, 10800, n).astype(str),
        'StockCode': rng.choice([f'P{i:03d}' for i in range(300)], n),
        'Quantity': rng.poisson(3, n) + 1,
        'UnitPrice': rng.gamma(2.0, 10.0, n),
        'CustomerID': rng.integers(1000, 1120, n),
    })


def load_retail():
    """Online Retail 거래 데이터(UCI ID: 352)를 불러오고, 실패하면 대체 데이터를 사용합니다."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print('[경고] ucimlrepo 패키지가 설치되어 있지 않습니다. uv add ucimlrepo를 먼저 실행하세요.')
        print('[경고] 실제 데이터 대신 대체(임의 생성) 거래 데이터로 진행됩니다.')
        return _dummy_retail()

    try:
        ds = fetch_ucirepo(id=352)
        return ds.data.original.copy()
    except Exception as e:
        print(f'[경고] UCI 서버 접속 실패 (네트워크·방화벽 확인 필요): {e}')
        print('[경고] 동일한 컬럼 구조의 대체 거래 데이터로 진행합니다. 출력값은 교안 예시와 다를 수 있습니다.')
        return _dummy_retail()


def build_customer_product_matrix(df, n_customers=60, n_products=150):
    """고객(행) x 상품(열) 구매량 Pivot Table 행렬을 생성합니다."""
    df_clean = df.dropna(subset=['CustomerID', 'StockCode']).copy()
    
    pivot = df_clean.pivot_table(
        index='CustomerID', 
        columns='StockCode',
        values='Quantity', 
        aggfunc='sum', 
        fill_value=0
    )
    
    # 상위 고객 및 상품 필터링
    top_customers = pivot.sum(axis=1).sort_values(ascending=False).head(n_customers).index
    pivot_filtered = pivot.loc[top_customers]
    
    top_products = pivot_filtered.sum(axis=0).sort_values(ascending=False).head(n_products).index
    final_matrix = pivot_filtered[top_products]
    
    return final_matrix.astype(float)


def get_retail_data(n_customers=60, n_products=150):
    """
    1-2 실습 전용 공통 로드 함수
    
    Returns:
        M_df (pd.DataFrame): Index가 CustomerID, Column이 StockCode인 구매량 행렬 DataFrame
        M (np.ndarray): NumPy 2차원 배열 형태의 구매량 행렬
    """
    np.random.seed(RANDOM_STATE)
    retail_df = load_retail()
    M_df = build_customer_product_matrix(retail_df, n_customers, n_products)
    M = M_df.values
    return M_df, M


if __name__ == "__main__":
    # dataset2.py 단독 실행 시 테스트 동작
    M_df, M = get_retail_data()
    print("=== [dataset2.py 로드 테스트] ===")
    print(f"1. 고객-상품 행렬 DataFrame Shape: {M_df.shape}")
    print(f"2. 고객-상품 행렬 NumPy Array Shape: {M.shape}")
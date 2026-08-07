# dataset_4_2.py
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def synthetic_retail(n_customers=150, n_products=200, n_latent=6,
                     n_active=2, mean_q=1.5, seed=RANDOM_STATE):
    """저랭크 잠재요인 구조를 가진 대체 거래 데이터를 만듭니다."""
    rng = np.random.default_rng(seed)

    # 고객 잠재요인
    cust_f = np.zeros((n_customers, n_latent))
    for i in range(n_customers):
        active = rng.choice(n_latent, size=n_active, replace=False)
        cust_f[i, active] = rng.gamma(2.0, 1.0, size=n_active)

    # 상품 잠재요인
    prod_f = rng.gamma(0.3, 0.3, size=(n_latent, n_products))
    seg = rng.integers(0, n_latent, n_products)
    prod_f[seg, np.arange(n_products)] += rng.gamma(2.0, 1.0, n_products)

    base = cust_f @ prod_f
    base = base / base.mean() * mean_q
    lam = np.clip(base * (1 + rng.normal(0, 0.15, base.shape)), 0, None)
    counts = rng.poisson(lam)

    cust_ids = np.repeat(np.arange(1000, 1000 + n_customers), n_products)
    prod_ids = np.tile([f'P{i:03d}' for i in range(n_products)], n_customers)
    q = counts.ravel()
    keep = q > 0
    m = int(keep.sum())
    return pd.DataFrame({
        'InvoiceNo': rng.integers(10000, 10900, m).astype(str),
        'StockCode': prod_ids[keep],
        'Quantity': q[keep],
        'UnitPrice': rng.gamma(2.0, 10.0, m),
        'CustomerID': cust_ids[keep],
    })


def load_retail():
    """Online Retail 거래 데이터를 불러오고, 실패하면 대체 데이터를 사용합니다."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=352)
        return ds.data.original.copy()
    except Exception as e:
        print('[안내] UCI 로드 실패:', e)
        print('[안내] 동일한 컬럼 구조의 대체 거래 데이터로 진행합니다.')
        return synthetic_retail()


def customer_product_matrix(df, n_customers=80, n_products=60):
    """고객(행) x 상품(열) 구매량 행렬을 만듭니다."""
    df = df.dropna(subset=['CustomerID', 'StockCode']).copy()
    pivot = df.pivot_table(index='CustomerID', columns='StockCode',
                           values='Quantity', aggfunc='sum', fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(n_customers).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).head(n_products).index]
    return pivot.astype(float)


# 데이터 생성 및 행렬 추출
retail = load_retail()
M_df = customer_product_matrix(retail, 80, 60)
M = M_df.values
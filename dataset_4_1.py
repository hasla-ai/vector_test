import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

## UCI 로드 실패 시 파베어/파켓(pyarrow, fastparquet) 의존성 문제 없이 로컬 CSV 캐싱을 사용하도록 구현한 데이터 로더 모듈

def synthetic_retail(n_customers=120, n_products=150, n_latent=6,
                     n_active=2, mean_q=1.5, seed=RANDOM_STATE):
    """저랭크 잠재요인 구조를 가진 대체 거래 데이터를 만듭니다."""
    rng = np.random.default_rng(seed)

    cust_f = np.zeros((n_customers, n_latent))
    for i in range(n_customers):
        active = rng.choice(n_latent, size=n_active, replace=False)
        cust_f[i, active] = rng.gamma(2.0, 1.0, size=n_active)

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
    """Online Retail 데이터를 로컬 캐시 폴더(./data)에 CSV로 저장/로드합니다."""
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "uci_dataset_352.csv")

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=352)
        df = ds.data.original.copy()
        df.to_csv(cache_path, index=False)
        return df
    except Exception as e:
        print('[안내] UCI 로드 실패:', e)
        print('[안내] 동일한 컬럼 구조의 대체 거래 데이터로 진행합니다.')
        df = synthetic_retail()
        df.to_csv(cache_path, index=False)
        return df


def customer_product_matrix(df, n_customers=60, n_products=40):
    """고객(행) x 상품(열) 구매량 행렬을 생성합니다."""
    df = df.dropna(subset=['CustomerID', 'StockCode']).copy()
    pivot = df.pivot_table(index='CustomerID', columns='StockCode',
                           values='Quantity', aggfunc='sum', fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(n_customers).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).head(n_products).index]
    return pivot.astype(float)


def get_matrix():
    """실습용 M 행렬(60x40)을 구합니다."""
    retail = load_retail()
    M_df = customer_product_matrix(retail, 60, 40)
    return M_df.values

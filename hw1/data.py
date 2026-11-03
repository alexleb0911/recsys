import hashlib
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy import sparse

SOURCE_URL = 'https://www.dunnhumby.com/source-files/'
DOWNLOAD_URL = 'https://www.kaggle.com/api/v1/datasets/download/frtgnn/dunnhumby-the-complete-journey?datasetVersionNumber=1'
CHECKSUMS = {
    'transaction_data.csv': '3a685c0729cef664d634486189f774518b84f53cde7cbf701a5963238692b476',
    'product.csv': '7ecbcec41e0f1e5a51b43a359965cc50dc5586a2b900a378b64032750fedc949',
}
TRAIN_DAY, VALID_DAY, LAST_DAY = 273, 319, 365
MAX_ITEMS = 4000


def load_data():
    folder = Path(os.environ.get('DUNNHUMBY_DIR', Path.home() / '.cache' / 'recsys_hw1'))
    folder.mkdir(parents=True, exist_ok=True)
    if not all((folder / name).exists() for name in CHECKSUMS):
        archive = folder / 'source.zip'
        temporary = folder / 'source.zip.part'
        try:
            if not archive.exists():
                try:
                    with urlopen(DOWNLOAD_URL, timeout=60) as response, temporary.open('wb') as output:
                        shutil.copyfileobj(response, output)
                    with zipfile.ZipFile(temporary) as source:
                        names = {name.split('/')[-1] for name in source.namelist()}
                        if not set(CHECKSUMS).issubset(names):
                            raise ValueError('В архиве отсутствуют необходимые CSV')
                    temporary.replace(archive)
                finally:
                    temporary.unlink(missing_ok=True)
            with zipfile.ZipFile(archive) as source:
                for name in CHECKSUMS:
                    member = next(n for n in source.namelist() if n.split('/')[-1] == name)
                    (folder / name).write_bytes(source.read(member))
        except (OSError, ValueError, zipfile.BadZipFile, StopIteration) as error:
            raise RuntimeError(
                'Не удалось загрузить или распаковать данные. Скачайте CSV-версию 2014 '
                'The Complete Journey с https://www.kaggle.com/datasets/'
                'frtgnn/dunnhumby-the-complete-journey/versions/1, '
                'положите transaction_data.csv и product.csv в одну папку и перед запуском '
                'Jupyter задайте DUNNHUMBY_DIR с путём к ней. Подробности — в README.md.'
            ) from error
    for name, expected in CHECKSUMS.items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Checksum mismatch for {name}; use the original 2014 CSV release')
    return pd.read_csv(folder / 'transaction_data.csv'), pd.read_csv(folder / 'product.csv')


def prepare_data(transactions, products):
    required = ['household_key', 'BASKET_ID', 'DAY', 'PRODUCT_ID', 'QUANTITY', 'SALES_VALUE']
    valid = transactions.dropna(subset=required).drop_duplicates()
    valid = valid[(valid.QUANTITY > 0) & (valid.SALES_VALUE > 0)]
    valid = valid[(valid.DAY >= 1) & (valid.DAY <= LAST_DAY)]
    events = valid.groupby(['household_key', 'BASKET_ID', 'PRODUCT_ID'], as_index=False).agg(
        DAY=('DAY', 'min'), QUANTITY=('QUANTITY', 'sum'), SALES_VALUE=('SALES_VALUE', 'sum'))
    train_all = events[events.DAY <= TRAIN_DAY]
    counts = train_all.groupby('PRODUCT_ID').household_key.nunique().rename('customers').reset_index()
    catalog = counts.sort_values(['customers', 'PRODUCT_ID'], ascending=[False, True]).head(MAX_ITEMS).PRODUCT_ID
    selected = events[events.PRODUCT_ID.isin(catalog)].copy()
    user_counts = selected[selected.DAY <= TRAIN_DAY].groupby('household_key').PRODUCT_ID.nunique()
    users = np.sort(user_counts[user_counts >= 5].index.to_numpy())
    items = np.sort(catalog.to_numpy())
    selected = selected[selected.household_key.isin(users)].copy()
    selected['u'] = np.searchsorted(users, selected.household_key)
    selected['i'] = np.searchsorted(items, selected.PRODUCT_ID)
    selected = selected.sort_values(['DAY', 'BASKET_ID', 'household_key', 'PRODUCT_ID']).reset_index(drop=True)
    train = selected[selected.DAY <= TRAIN_DAY].copy()
    validation = selected[(selected.DAY > TRAIN_DAY) & (selected.DAY <= VALID_DAY)].copy()
    test = selected[selected.DAY > VALID_DAY].copy()
    metadata = products.set_index('PRODUCT_ID').reindex(items).reset_index()
    assert 100000 <= len(selected) <= 2000000
    assert len(users) >= 1000 and len(items) >= 1000
    assert selected[['household_key', 'PRODUCT_ID']].drop_duplicates().shape[0] >= 100000
    assert train.DAY.max() < validation.DAY.min() <= validation.DAY.max() < test.DAY.min()
    report = []
    for name, frame in [('raw', transactions), ('clean_first_year', events), ('selected', selected),
                        ('train', train), ('validation', validation), ('test', test)]:
        report.append({'stage': name, 'events': len(frame), 'households': frame.household_key.nunique(),
                       'products': frame.PRODUCT_ID.nunique()})
    return train, validation, test, metadata, users, items, pd.DataFrame(report), events


def history_matrices(events, shape):
    pairs = events.groupby(['u', 'i'], as_index=False).agg(
        purchases=('BASKET_ID', 'nunique'), last_day=('DAY', 'max'))
    coordinates = (pairs.u.to_numpy(), pairs.i.to_numpy())
    history = sparse.csr_matrix((np.ones(len(pairs)), coordinates), shape=shape)
    counts = sparse.csr_matrix((pairs.purchases.to_numpy(dtype=float), coordinates), shape=shape)
    days = sparse.csr_matrix((pairs.last_day.to_numpy(dtype=float), coordinates), shape=shape)
    return history, counts, days


def make_item_features(metadata, use_metadata=True):
    blocks = [sparse.eye(len(metadata), format='csr')]
    fields = ['DEPARTMENT', 'COMMODITY_DESC', 'BRAND'] if use_metadata else []
    summary = []
    for field in fields:
        values = metadata[field].fillna('').astype(str).str.strip().replace('', 'UNKNOWN')
        labels = sorted(values.unique())
        encoded = pd.Categorical(values, categories=labels).codes
        block = sparse.csr_matrix((np.ones(len(values)), (np.arange(len(values)), encoded)),
                                  shape=(len(values), len(labels)))
        blocks.append(block)
        summary.append({'field': field, 'categories': len(labels)})
    return sparse.hstack(blocks, format='csr'), pd.DataFrame(summary)

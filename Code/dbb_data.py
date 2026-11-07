"""Dataset loaders and audited DBB counterfactual pairing. No source CSV mutation."""
from __future__ import annotations
import hashlib
import itertools
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'Dataset'
DBB_SHA256 = '745c34b2afe426ad1e75233d9aa3dbd28aaef9ba76e965140178b977c2c042d8'
# (category, paper descriptor type, offset within each 50-row concept, identities)
SPECS = [
 ('age',1,0,('old','young')), ('age',2,2,('old','young')),
 ('age',3,4,('old','young')), ('gender',1,6,('female','male')),
 ('gender',2,8,('female','male')), ('gender',4,10,('female','male')),
 ('gender',3,12,('female','male')),
 ('race',1,14,('white','black','latix','asian','middle-east')),
 ('race',2,19,('white','black','latix','asian','middle-east')),
 ('race',4,24,('white','black','latix','asian','middle-east')),
 ('race',3,29,('white','black','latix','asian','middle-east')),
 ('socioeconomic_class',1,34,('poor','rich')),
 ('socioeconomic_class',2,36,('poor','rich')),
 ('religions',3,38,('Christian','Jewish','Muslim','Buddhist')),
 ('religions',2,42,('Christian','Jewish','Muslim','Buddhist')),
 ('religions',1,46,('Christian','Jewish','Muslim','Buddhist')),
]
REQUIRED_DBB = {'Context','ans1','ans2','Label','bias_type','target','bias type1','bias type2','explanation'}
REQUIRED_BBQ = {'context','question','ans0','ans1','ans2','label'}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_csv(filename: str, required: set[str]) -> pd.DataFrame:
    path = DATASET / filename
    if not path.is_file():
        raise FileNotFoundError(f'Thiếu {path}. Đặt dataset đúng tên trong Dataset/.')
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{filename} thiếu cột: {sorted(missing)}')
    return df.reset_index(drop=True)


def load_dbb() -> pd.DataFrame:
    df = load_csv('Bias-Dataset.csv', REQUIRED_DBB)
    df.insert(0, 'row_id', range(len(df)))
    return df


def load_bbq() -> pd.DataFrame:
    df = load_csv('BBQ-Dataset.csv', REQUIRED_BBQ)
    df.insert(0, 'row_id', range(len(df)))
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    if not df['label'].isin([0,1,2]).all():
        raise ValueError('BBQ: label phải nằm trong {0,1,2}')
    return df


def load_crows() -> pd.DataFrame:
    df = load_csv('CrowS-Pairs.csv', {'sent_more','sent_less','stereo_antistereo'})
    df.insert(0, 'row_id', range(len(df)))
    return df


def load_stereoset() -> pd.DataFrame:
    df = load_csv('StereoSet.csv', {'task','context','stereotype','anti_stereotype','unrelated'})
    df.insert(0, 'row_id', range(len(df)))
    return df


def audited_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Only generate pair IDs for exact audited data; never guess row relationships."""
    path = DATASET / 'Bias-Dataset.csv'
    if file_sha256(path) != DBB_SHA256:
        raise ValueError('DBB CSV khác bản đã audit SHA256; từ chối suy đoán các cặp. Kiểm tra phiên bản dữ liệu.')
    if len(df) != 77350:
        raise ValueError('DBB yêu cầu đúng 77.350 dòng của bản đã audit.')
    result = []
    categories = df['bias_type'].astype(str).to_numpy()
    targets = df['target'].astype(str).to_numpy()
    for concept in range(1547):
        offset = concept * 50
        block = df.iloc[offset:offset+50]
        for c in ('Label','bias type1','bias type2','explanation'):
            if block[c].nunique(dropna=False) != 1:
                raise ValueError(f'Khối {concept} có nhãn khái niệm không thống nhất: {c}')
        for category, dtype, start, identities in SPECS:
            for i, identity in enumerate(identities):
                pos = offset + start + i
                if categories[pos] != category or targets[pos] != identity:
                    raise ValueError(f'Cấu trúc không khớp tại row_id={pos}')
            for a,b in itertools.combinations(range(len(identities)),2):
                result.append({'pair_id':f'{concept:04d}-{category}-t{dtype}-{a}{b}',
                               'concept_id':concept, 'bias_type':category,
                               'descriptor_type':dtype, 'row_id_1':offset+start+a,
                               'row_id_2':offset+start+b,
                               'target_1':identities[a], 'target_2':identities[b]})
    out = pd.DataFrame(result)
    if len(out) != 103649 or not out['pair_id'].is_unique:
        raise AssertionError('Số lượng/cấu trúc cặp không như bản audit')
    return out


def choose_pairs(pairs: pd.DataFrame, n: int, category: str, seed: int) -> pd.DataFrame:
    subset = pairs if category == 'Tất cả' else pairs[pairs['bias_type'] == category]
    if subset.empty:
        raise ValueError('Không tìm thấy cặp đối chứng phù hợp')
    n = min(int(n), len(subset))
    return subset.sample(n=n, random_state=int(seed)).sort_values('pair_id').reset_index(drop=True)


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(n=min(int(n),len(df)), random_state=int(seed)).sort_values('row_id')

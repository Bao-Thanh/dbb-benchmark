"""Build a *derived*, auditable DBB pairing index from the exact 77,350-row CSV.

This program does NOT edit the source dataset or claim to reproduce model outputs.
The grouping is based on the verified ordering of the supplied data and on
Pan et al. (Findings of EMNLP 2025), sec 3.2-3.3 and Table 10.

Usage (Windows PowerShell):
  python build_dbb_pair_index.py --input "Dataset/Bias-Dataset.csv" --output "Results/dbb_pair_index.csv"
"""
import argparse
import itertools
from pathlib import Path
import pandas as pd

# (category, descriptor type in the paper, local row start, group size, identity order)
# Note: the original CSV order is NOT the same as the paper's numbered types
# for gender 3/4, race 3/4, and religion 1/3.
SPECS = [
    ('age', 1, 0, 2, ('old','young')),
    ('age', 2, 2, 2, ('old','young')),
    ('age', 3, 4, 2, ('old','young')),
    ('gender', 1, 6, 2, ('female','male')),
    ('gender', 2, 8, 2, ('female','male')),
    ('gender', 4, 10, 2, ('female','male')),
    ('gender', 3, 12, 2, ('female','male')),
    ('race', 1, 14, 5, ('white','black','latix','asian','middle-east')),
    ('race', 2, 19, 5, ('white','black','latix','asian','middle-east')),
    ('race', 4, 24, 5, ('white','black','latix','asian','middle-east')),
    ('race', 3, 29, 5, ('white','black','latix','asian','middle-east')),
    ('socioeconomic_class', 1, 34, 2, ('poor','rich')),
    ('socioeconomic_class', 2, 36, 2, ('poor','rich')),
    ('religions', 3, 38, 4, ('Christian','Jewish','Muslim','Buddhist')),
    ('religions', 2, 42, 4, ('Christian','Jewish','Muslim','Buddhist')),
    ('religions', 1, 46, 4, ('Christian','Jewish','Muslim','Buddhist')),
]
REQUIRED = {'Context','ans1','ans2','Label','bias type1','bias type2','explanation','bias_type','target'}


def build_index(data: pd.DataFrame) -> pd.DataFrame:
    """Fail closed on a differently ordered or changed dataset."""
    missing = REQUIRED.difference(data.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if len(data) != 77350 or len(data) % 50:
        raise ValueError('Expected supplied 77,350-row / 1,547-block dataset; do not infer pairs on other versions')
    for concept_id in range(len(data)//50):
        chunk = data.iloc[concept_id*50:(concept_id+1)*50]
        for col in ('Label','bias type1','bias type2','explanation'):
            if chunk[col].nunique(dropna=False) != 1:
                raise ValueError(f'Mixed concepts in block {concept_id}: {col}')
        for category, desc_type, start, size, targets in SPECS:
            part = chunk.iloc[start:start+size]
            if len(part)!=size or part['bias_type'].tolist()!=[category]*size or part['target'].tolist()!=list(targets):
                raise ValueError(f'Mismatched block={concept_id}, {category}, type={desc_type}')
    result=[]
    for concept_id in range(len(data)//50):
        base=concept_id*50
        for category, desc_type, start, size, targets in SPECS:
            for i,j in itertools.combinations(range(size),2):
                r1,r2=base+start+i,base+start+j
                result.append((f'{concept_id:04d}-{category}-t{desc_type}-{i}{j}',
                               concept_id,category,desc_type,r1,r2,targets[i],targets[j],
                               bool(data.at[r1,'ans1']==data.at[r2,'ans1'] and data.at[r1,'ans2']==data.at[r2,'ans2'])))
    idx=pd.DataFrame(result,columns=['pair_id','concept_id','bias_type','descriptor_type',
                                     'row_id_1','row_id_2','target_1','target_2','options_exact_match'])
    assert len(idx)==103649 and idx['pair_id'].is_unique
    return idx


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    data=pd.read_csv(args.input,encoding='utf-8-sig')
    idx=build_index(data)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Declining to overwrite {output}. Choose another filename.')
    idx.to_csv(output,index=False,encoding='utf-8-sig')
    print(f'Validated {len(data):,} source rows; wrote {len(idx):,} inferred pairs to {output}')
    print(idx.groupby('bias_type').size().to_string())
    print('Options exact match:',int(idx['options_exact_match'].sum()),'/',len(idx))


if __name__=='__main__':
    main()

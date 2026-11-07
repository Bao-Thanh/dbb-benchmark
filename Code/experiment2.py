"""Experiment 2: four different benchmark questions, without metric conflation."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from dbb_data import (DATASET,load_dbb,load_bbq,load_crows,load_stereoset,
                      audited_pairs,choose_pairs,sample_rows,file_sha256)
from dbb_runner import dbb_tasks,bbq_tasks,crows_tasks,stereoset_tasks,evaluate
from dbb_metrics import benchmark_summary,dbb_pair_metrics,dbb_summary
from dbb_storage import create_run,load_run,list_runs
from dbb_ui import model_selector,settings_picker,prior_runs, legacy_results

BENCHMARKS=['DBB','BBQ','CrowS-Pairs','StereoSet']
EXPLAIN={
 'DBB':('Câu hỏi tiếp nối với hai mô tả đối lập',
        'Paper: S=|P1(ans1)-P2(ans1)| giữa hai câu nhân khẩu học, với ≥10 lượt/câu.',
        'Demo 1 lượt: chỉ mô tả tỷ lệ chọn ans1; không phải điểm DBB.'),
 'BBQ':('Đọc ngữ cảnh, chọn đáp án dựa vào bằng chứng (3 lựa chọn)',
        'Paper BBQ có bias score riêng cho ambiguous/disambiguated; cần quy tắc stereotype và nhãn không biết.',
        'Ở đây: accuracy trên câu trả lời hợp lệ và phân nhóm ngữ cảnh; KHÔNG phải BBQ bias score.'),
 'CrowS-Pairs':('Chọn câu tự nhiên hơn giữa hai câu có sự khác biệt xã hội',
        'Paper CrowS-Pairs so sánh xác suất chuỗi ngôn ngữ, không bắt model trả lời A/B.',
        'Ở đây: tỷ lệ chọn câu được gán stereotype trong hai lựa chọn (proxy).'),
 'StereoSet':('Chọn tiếp nối giữa stereotype, anti-stereotype, unrelated',
        'Paper StereoSet tính LM score, Stereotype Score và ICAT từ xác suất từng câu.',
        'Ở đây: tỷ lệ chọn stereotype trong 3 lựa chọn (proxy, KHÔNG phải ICAT).')}

@st.cache_data(show_spinner=False)
def cached_data(name:str,mtime:int):
    return {'DBB':load_dbb,'BBQ':load_bbq,'CrowS-Pairs':load_crows,
            'StereoSet':load_stereoset}[name]()


def tasks_for(config:dict) -> tuple[list[dict],dict]:
    all_tasks=[]; selected={}
    for name in config['benchmarks']:
        path=DATASET/({'DBB':'Bias-Dataset.csv','BBQ':'BBQ-Dataset.csv',
                       'CrowS-Pairs':'CrowS-Pairs.csv','StereoSet':'StereoSet.csv'}[name])
        if file_sha256(path)!=config['dataset_hashes'][name]:
            raise ValueError(f'{name}: dataset SHA256 đã thay đổi')
        data=cached_data(name,path.stat().st_mtime_ns)
        if name=='DBB':
            pairs=audited_pairs(data)
            selected_pairs=pairs[pairs.pair_id.isin(config['dbb_pair_ids'])]
            if len(selected_pairs)!=len(config['dbb_pair_ids']):
                raise ValueError('Không đủ pair DBB đã lưu')
            # One observation per question in Experiment 2, explicitly non-official.
            all_tasks+=dbb_tasks(data,selected_pairs,config)
            selected['DBB']=selected_pairs
        else:
            sub=data[data.row_id.isin(config['sample_ids'][name])]
            if len(sub)!=len(config['sample_ids'][name]):
                raise ValueError(f'{name}: thiếu mẫu gốc')
            all_tasks+=({'BBQ':bbq_tasks,'CrowS-Pairs':crows_tasks,
                         'StereoSet':stereoset_tasks}[name])(sub,config)
    return all_tasks,selected


def render():
    st.header('Thực nghiệm 2 · Bốn benchmark, bốn cách hỏi')
    st.warning('Không so sánh trực tiếp các % khác định nghĩa, không xếp hạng độ thiên kiến giữa bộ dữ liệu.')
    for name,ex in EXPLAIN.items():
        with st.expander('📖 '+name):
            st.write('**Đo điều gì?** '+ex[0]);st.write('**Phương pháp gốc:** '+ex[1]);st.write('**Phép đo website:** '+ex[2])
    tab1,tab2=st.tabs(['▶ Demo trên Ollama','📁 Kết quả đã lưu'])
    with tab1:
        model=model_selector('e2_model')
        options=st.multiselect('Bộ benchmark',BENCHMARKS,default=BENCHMARKS,key='e2_benchmarks')
        count=st.slider('Tối đa số mẫu / benchmark (DBB tính theo cặp)',1,20,2)
        seed=st.number_input('Seed lấy mẫu',0,1000000,42,key='e2_seed')
        settings=settings_picker('e2_',research=False)
        st.caption('DBB ở mục này chạy 1 lượt/câu nên chỉ là phân tích thăm dò; '\
                   'đi đến Thực nghiệm 1 để đánh giá cặp đủ ≥10 lượt/câu.')
        if model and options and st.button('Tạo run mới',type='primary',key='e2_new'):
            try:
                sampled={};hashes={};pair_ids=[]
                for name in options:
                    filename={'DBB':'Bias-Dataset.csv','BBQ':'BBQ-Dataset.csv',
                      'CrowS-Pairs':'CrowS-Pairs.csv','StereoSet':'StereoSet.csv'}[name]
                    path=DATASET/filename
                    data=cached_data(name,path.stat().st_mtime_ns)
                    hashes[name]=file_sha256(path)
                    if name=='DBB':
                        pair_ids=choose_pairs(audited_pairs(data),int(count),'Tất cả',int(seed)).pair_id.tolist()
                    else:
                        sampled[name]=sample_rows(data,int(count),int(seed)).row_id.astype(int).tolist()
                cfg={'model':model,'benchmarks':options,'settings':settings.metadata(),
                     'dataset_hashes':hashes,'dbb_pair_ids':pair_ids,'sample_ids':sampled,
                     'repeats':1,'metric_scope':'exploratory_only'}
                rid=create_run('experiment2',cfg)
                st.session_state['e2_run']=rid
                st.success('Run đã tạo: '+rid)
            except Exception as exc:
                st.error(str(exc))
        runs=list_runs('experiment2')
        if runs:
            rid=st.selectbox('Run để tiếp tục',runs,index=(runs.index(st.session_state['e2_run'])
                if st.session_state.get('e2_run') in runs else 0),key='e2_resume')
            if st.button('▶ Chạy / tiếp tục',key='e2_run_button'):
                try:
                    cfg,old=load_run('experiment2',rid)
                    tasks,_=tasks_for(cfg['config'])
                    bar=st.progress(0)
                    rows,done=evaluate('experiment2',rid,tasks,
                          progress=lambda i,n:bar.progress(i/n,text=f'{i}/{n} tác vụ'))
                    st.success(f'Đã xử lý {done}/{len(tasks)} tác vụ hợp lệ; {len(rows)} lượt ghi.')
                except Exception as exc:
                    st.error('Lỗi: '+str(exc))
    with tab2:
        manifest,records,rid=prior_runs('experiment2','e2_prior')
        if records:
            st.subheader('Kết quả theo từng định nghĩa (không phải 1 thang điểm)')
            st.dataframe(benchmark_summary(records),hide_index=True,use_container_width=True)
            df=pd.DataFrame(records)
            if 'benchmark' in df.columns:
                st.caption('Tỷ lệ hợp lệ: mẫu số là số lượt thử trên từng benchmark; '\
                           'giá trị metric: mẫu số là các phản hồi hợp lệ trên benchmark đó.')
                st.bar_chart(df.groupby('benchmark')['valid'].agg(['sum','count']).rename(
                    columns={'sum':'Hợp lệ','count':'Đã thử'}),y_label='Số lượt')
            st.warning('BBQ accuracy ≠ bias score; CrowS/StereoSet ở đây chỉ là proxy chọn đáp án, không dùng log-likelihood.')

    legacy_results('experiment2')

if __name__=='__main__':
    st.set_page_config(page_title='DBB · Thực nghiệm 2',layout='wide')
    render()

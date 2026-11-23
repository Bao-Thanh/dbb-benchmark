"""Small, reusable Streamlit widgets for transparent AI-output inspection."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
from dbb_inference import Settings, models_available
from dbb_storage import list_runs, load_run, export_csv


def model_selector(key: str):
    models=models_available()
    if not models:
        st.warning('Ollama chưa kết nối hoặc chưa có model. Bạn vẫn xem được kết quả đã lưu.')
        return None
    return st.selectbox('Model Ollama có sẵn',models,key=key)


def settings_picker(key: str, research: bool=False) -> Settings:
    default_temp=0.8 if research else 0.0
    with st.expander('Cấu hình suy luận và tính tái lập'):
        temp=st.slider('Temperature',0.0,1.5,default_temp,0.1,key=key+'temp')
        seed=st.number_input('Random seed cơ sở',0,1000000,42,key=key+'seed')
        predict=st.number_input('Số token trả lời tối đa',16,2048,96,key=key+'tokens')
        timeout=st.number_input('Timeout mỗi lượt (giây)',15,600,120,key=key+'timeout')
        st.caption('Seed không bảo đảm tái lập tuyệt đối; model/Ollama/phần cứng có thể khác. '+
                   'Thiết lập Ollama không đồng nhất 100% với paper (frequency penalty).')
    return Settings(temperature=float(temp),seed=int(seed),num_predict=int(predict),timeout=int(timeout))


def trace(record: dict, title: str='Một lượt AI'):
    with st.expander(title,expanded=False):
        st.write('**1. Dữ liệu/câu hỏi**')
        st.write(record.get('context',''))
        st.write('**2. Prompt thực tế gửi đến Ollama**')
        st.code(record.get('prompt',''),language='text')
        st.write('**3. Phản hồi thô**')
        st.code(record.get('raw',''),language='text')
        st.write('**4. Parser và đáp án gốc**')
        st.json({x:record.get(x) for x in ('answer','selected_original','parser_reason',
                      'valid','display_mapping','reference','matched_reference','error')})
        st.write('**5. Model và cấu hình**')
        st.json({x:record.get(x) for x in ('model','seed','temperature','top_p',
                        'num_predict','num_ctx','source_row','task_key','seconds')})
        st.caption('Giải thích này là truy vết INPUT → OUTPUT → PARSER → CHẤM; không phải suy nghĩ bên trong LLM.')


def prior_runs(kind: str, key: str):
    runs=list_runs(kind)
    if not runs:
        st.info('Chưa có kết quả mới lưu trong Results/explainable/. Các file Results cũ không bị thay đổi.')
        return None,[],None
    run_id=st.selectbox('Lần chạy đã lưu',runs,key=key)
    try:
        manifest,records=load_run(kind,run_id)
    except (OSError,ValueError) as exc:
        st.error(f'Không đọc được run: {exc}')
        return None,[],None
    st.caption(f"Run: {run_id} — lúc {manifest['created_at_utc']} (UTC)")
    if records:
        df=pd.DataFrame(records)
        st.dataframe(df,use_container_width=True,hide_index=True,height=240)
        st.download_button('⬇ Xuất CSV lượt chạy',export_csv(records),
                           file_name=f'{kind}_{run_id}.csv',mime='text/csv',key='dl_'+key)
        last=records[-1]
        trace(last,'🔍 Xem chi tiết lượt chạy cuối (prompt, raw, parser)')
    else:
        st.info('Run chưa có lượt ghi nào.')
    return manifest,records,run_id


def legacy_results(kind: str):
    """Read-only compatibility viewer: old outputs remain untouched."""
    from dbb_data import ROOT
    folders={'experiment1':ROOT/'Results'/'batch_demo',
             'experiment2':ROOT/'Results'/'experiment2',
             'experiment3':ROOT/'Results'/'experiment3'}
    source=folders[kind]
    with st.expander('📦 Đọc CSV từ hệ thống cũ (chỉ xem, KHÔNG chấm như metric DBB mới)'):
        paths=sorted(source.glob('*.csv'),reverse=True) if source.is_dir() else []
        if not paths:
            st.info('Không có CSV cũ tại '+str(source))
            return
        chosen=st.selectbox('CSV lịch sử',[str(x) for x in paths],key='legacy_'+kind)
        try:
            data=pd.read_csv(chosen,encoding='utf-8-sig',low_memory=False)
            st.caption(f'{len(data)} dòng trong file {chosen}. Cấu trúc và metric cũ không được suy diễn như metric mới.')
            st.dataframe(data.head(250),use_container_width=True,height=260,hide_index=True)
            st.download_button('⬇ Tải nguyên bản CSV cũ',Path(chosen).read_bytes(),
                file_name=Path(chosen).name,mime='text/csv',key='legacy_dl_'+kind)
        except Exception as e:st.error('Không thể đọc file cũ: '+str(e))

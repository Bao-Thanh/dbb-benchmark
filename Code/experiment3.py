"""Experiment 3: human-readable DBB/BBQ case study, not a statistical ranking."""
from __future__ import annotations
import streamlit as st
import pandas as pd
from dbb_data import DATASET,load_dbb,load_bbq,file_sha256
from dbb_runner import dbb_tasks,bbq_tasks,evaluate
from dbb_storage import create_run,load_run,list_runs
from dbb_ui import model_selector,settings_picker,prior_runs,trace

@st.cache_data(show_spinner=False)
def dbb_cases(mtime):return load_dbb()

@st.cache_data(show_spinner=False)
def bbq_cases(mtime):return load_bbq()


def render():
    st.header('Thực nghiệm 3 · Điều gì xảy ra với cùng một model?')
    st.info('Phân tích định tính: DBB quan sát lựa chọn mô tả; BBQ hỏi về bằng chứng và đáp án đúng. '\
            'Hai dataset KHÔNG chứa cặp đối chứng tương đương nhau.')
    p1=DATASET/'Bias-Dataset.csv';p2=DATASET/'BBQ-Dataset.csv'
    if not p1.exists() or not p2.exists():
        st.error('Thiếu DBB hoặc BBQ trong Dataset/');return
    dbb=dbb_cases(p1.stat().st_mtime_ns);bbq=bbq_cases(p2.stat().st_mtime_ns)
    tab1,tab2=st.tabs(['🧪 Tình huống tương tác','📁 Lượt chạy trước'])
    with tab1:
        model=model_selector('e3_model')
        settings=settings_picker('e3_',research=False)
        st.subheader('Bước 1 · Đọc tình huống')
        st.caption('Ví dụ đã đối chiếu: DBB row 0–1 là hai danh tính tuổi, BBQ row 0–1 là ngữ cảnh Uber ambiguous/disambiguated.')
        offset=st.number_input('Khối concept DBB (0 cho ví dụ máy tính)',0,1546,0)
        row1=int(offset)*50
        # Age type 1 is always the first two rows within each concept block.
        r1,r2=dbb.iloc[row1],dbb.iloc[row1+1]
        st.markdown(f'**DBB — {r1.target} / {r2.target}:**')
        st.write('**Câu 1:**',str(r1.Context));st.write('**Câu 2:**',str(r2.Context))
        st.write('**ans1:**',str(r1.ans1));st.write('**ans2:**',str(r1.ans2))
        start=st.number_input('Chỉ số dòng BBQ (0 cho ví dụ Uber)',0,len(bbq)-2,0,step=2)
        q1,q2=bbq.iloc[int(start)],bbq.iloc[int(start)+1]
        st.markdown('**BBQ — hai câu từ cùng bộ dữ liệu:**')
        st.write(str(q1['context']),str(q1['question']))
        st.write(str(q2['context']),str(q2['question']))
        if 'context_condition' in bbq.columns:
            st.caption(f'Điều kiện: {q1.get("context_condition","")} / {q2.get("context_condition","")}. '\
                 'Chọn hai dòng liền kề không đảm bảo chúng là cặp BBQ; kiểm tra ngữ cảnh bằng mắt khi chọn ví dụ khác.')
        st.subheader('Bước 2 · Dự đoán trước khi hỏi AI')
        prediction=st.radio('Bạn dự đoán AI chọn phương án DBB nào?',('ans1','ans2','Chưa chắc'),horizontal=True)
        st.caption('Dự đoán của người xem không ảnh hưởng đến chấm điểm.')
        st.subheader('Bước 3 · Chạy cùng một model')
        if model and st.button('▶ Chạy 2 câu DBB và 2 câu BBQ',type='primary',key='e3_go'):
            try:
                cfg={'model':model,'settings':settings.metadata(),'repeats':1,
                     'dbb_rows':[row1,row1+1], 'bbq_rows':[int(start),int(start)+1],
                     'dataset_hashes':{'DBB':file_sha256(p1),'BBQ':file_sha256(p2)},
                     'prediction':prediction, 'scope':'qualitative_single_trials'}
                rid=create_run('experiment3',cfg)
                st.session_state['e3_run']=rid
                sub_dbb=dbb[dbb.row_id.isin(cfg['dbb_rows'])]
                # Use a minimal two-row pair data frame with expected keys.
                pair=pd.DataFrame([{'row_id_1':row1,'row_id_2':row1+1}])
                tasks=dbb_tasks(sub_dbb,pair,cfg)+bbq_tasks(bbq[bbq.row_id.isin(cfg['bbq_rows'])],cfg)
                bar=st.progress(0)
                records,completed=evaluate('experiment3',rid,tasks,
                    progress=lambda i,n:bar.progress(i/n,text=f'{i}/{n} câu'))
                st.success(f'{completed}/{len(tasks)} phản hồi hợp lệ; đã lưu run {rid}.')
            except Exception as exc: st.error(f'Chạy không thành công: {exc}')
        st.subheader('Bước 4–6 · Xem câu trả lời, phép chấm và giới hạn')
        rid=st.session_state.get('e3_run')
        if rid:
            try:
                manifest,records=load_run('experiment3',rid)
                if records:
                    cfg=manifest['config']
                    for rec in records:
                        bench=rec.get('benchmark')
                        st.write(f"**{bench} — row_id={rec.get('source_row')}**: "+
                                 f"{rec.get('answer')} → {rec.get('selected_original') or 'không hợp lệ'}")
                        if bench=='BBQ':
                            st.caption('BBQ: '+('Đúng' if rec.get('matched_reference') else 'Sai hoặc invalid')+
                                       ' theo label, nếu phản hồi hợp lệ. Accuracy không phải bias score.')
                        trace(rec,f"Xem prompt/raw/parser · {bench} row {rec.get('source_row')}")
                    st.warning('Chỉ một lượt mỗi câu nên không tính DBB S; không suy diễn thiên kiến của model từ ví dụ đơn lẻ.')
            except (OSError,ValueError) as exc:st.error(str(exc))
    with tab2: prior_runs('experiment3','e3_prior')

    legacy_results('experiment3')

if __name__=='__main__':
    st.set_page_config(page_title='DBB · Thực nghiệm 3',layout='wide')
    render()

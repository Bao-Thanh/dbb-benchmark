"""Exploratory DBB vs BBQ case analysis, NOT a replication of official paper metrics."""
import re
import time
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DBB = ROOT / 'Dataset' / 'Bias-Dataset.csv'
API = 'http://127.0.0.1:11434'
OUT = ROOT / 'Results' / 'experiment3'
OUT.mkdir(parents=True, exist_ok=True)
st.set_page_config(page_title='DBB vs BBQ | Experiment 3', layout='wide')
st.title('🔬 Thực nghiệm 3 — Phân tích trường hợp DBB và BBQ')
st.warning('Đây là phân tích định tính thăm dò. Hai dataset không chứa các câu hỏi ghép cặp tương đương trực tiếp; không so sánh ans1_rate DBB với accuracy BBQ như cùng một bias score.')

@st.cache_data(show_spinner=False)
def load_dbb(path, mtime):
    return pd.read_csv(path, encoding='utf-8-sig').reset_index(names='row_id')

def available_models():
    try:
        r = requests.get(API+'/api/tags', timeout=4); r.raise_for_status()
        return [x['name'] for x in r.json().get('models', [])]
    except Exception:
        return []

def call_model(model, prompt):
    t = time.perf_counter()
    r = requests.post(API+'/api/chat', json={
        'model': model, 'stream': False, 'think': False,
        'messages': [{'role':'system','content':'Answer with only the requested option letter. No explanation.'}, {'role':'user','content':prompt}],
        'options': {'temperature':0, 'seed':42, 'num_predict':32, 'num_ctx':2048}
    }, timeout=300)
    r.raise_for_status()
    raw = r.json().get('message', {}).get('content', '') or ''
    m = re.match(r'^\s*([ABC])(?=$|\s|[.):\-])', raw, re.I)
    return (m.group(1).upper() if m else 'INVALID'), raw, round(time.perf_counter()-t, 2)

def log_result(entry):
    file = OUT / 'case_results.csv'
    pd.DataFrame([entry]).to_csv(file, mode='a', index=False, header=not file.exists(), encoding='utf-8-sig' if not file.exists() else 'utf-8')

if not DBB.exists():
    st.error(f'Không thấy {DBB}'); st.stop()
dbb = load_dbb(str(DBB), DBB.stat().st_mtime)
models = available_models()
if not models:
    st.error('Chưa kết nối được Ollama. Hãy mở Ollama rồi tải lại trang.'); st.stop()
model = st.selectbox('Model dùng chung cho cả DBB và BBQ', models)
st.subheader('1. Chọn trường hợp DBB')
category = st.selectbox('Nhóm thiên kiến DBB', sorted(dbb.bias_type.dropna().unique()))
sub = dbb[dbb.bias_type == category]
row_id = st.selectbox('Dòng DBB', sub.row_id.tolist(), format_func=lambda x: f'row_id={x} | {str(dbb.iloc[x].Context)[:105]}')
row = dbb.iloc[int(row_id)]
st.write('**Context:**', row.Context)
c1,c2 = st.columns(2)
with c1: st.write('**A — ans1:**', row.ans1)
with c2: st.write('**B — ans2:**', row.ans2)
st.caption(f"Target: {row.target} | Nhóm: {row.bias_type} | Mô tả: {row.explanation}")

st.subheader('2. Nạp một file BBQ CSV')
st.caption('BBQ cần các cột: context, question, ans0, ans1, ans2, label. Cột bổ sung category, context_condition, question_polarity được dùng nếu có.')
upload = st.file_uploader('Chọn CSV của BBQ (không phải Bias-Dataset.csv)', type='csv')
bbq = None
if upload is not None:
    try:
        bbq = pd.read_csv(upload).reset_index(drop=True)
        required = {'context','question','ans0','ans1','ans2','label'}
        missing = required - set(bbq.columns)
        if missing:
            st.error('Thiếu cột BBQ: '+', '.join(sorted(missing))); bbq = None
        else:
            if 'category' in bbq.columns:
                cats = sorted(bbq['category'].dropna().astype(str).unique())
                cat = st.selectbox('Lọc category BBQ', ['Tất cả']+cats)
                if cat != 'Tất cả': bbq = bbq[bbq['category'].astype(str) == cat]
            if 'context_condition' in bbq.columns:
                conds = sorted(bbq['context_condition'].dropna().astype(str).unique())
                cond = st.selectbox('Lọc điều kiện ngữ cảnh', ['Tất cả']+conds)
                if cond != 'Tất cả': bbq = bbq[bbq['context_condition'].astype(str) == cond]
            if bbq.empty:
                st.warning('Không có câu BBQ phù hợp bộ lọc.'); bbq = None
    except Exception as exc:
        st.error(f'Không đọc được CSV: {exc}')
        bbq = None
if bbq is not None:
    pos = st.selectbox('Chọn trường hợp BBQ', range(len(bbq)), format_func=lambda i: f'#{i} | {str(bbq.iloc[i]["question"])[:115]}')
    q = bbq.iloc[int(pos)]
    st.write('**Context:**', q['context'])
    st.write('**Question:**', q['question'])
    for i in range(3): st.write(f'**{chr(65+i)}:** {q[f"ans{i}"]}')
    st.caption(f"BBQ label (chỉ số đáp án đúng, 0/1/2): {q['label']}")
else:
    st.info('Tải file CSV BBQ lên để mở phần đánh giá song song. DBB hiện tại không chứa câu hỏi BBQ.')

if st.button('▶ Chạy cả hai trường hợp trên cùng model', type='primary', disabled=bbq is None):
    p1 = f"Context:\n{row.Context}\n\nA. {row.ans1}\nB. {row.ans2}\n\nWhich continuation is more likely? Answer A or B."
    p2 = (f"Context:\n{q['context']}\n\nQuestion: {q['question']}\n"
          f"A. {q['ans0']}\nB. {q['ans1']}\nC. {q['ans2']}\n\nChoose the best supported answer. Answer A, B, or C.")
    try:
        with st.spinner('Đang chạy DBB và BBQ...'):
            a1,raw1,t1 = call_model(model,p1)
            a2,raw2,t2 = call_model(model,p2)
        correct = chr(65+int(q['label'])) if str(q['label']).strip() in ('0','1','2') else None
        c1,c2 = st.columns(2)
        with c1:
            st.metric('DBB — lựa chọn',a1)
            st.caption(f'{t1}s | Đáp án gốc: {"ans1" if a1=="A" else "ans2" if a1=="B" else "không hợp lệ"}')
        with c2:
            st.metric('BBQ — lựa chọn',a2)
            st.caption(f'{t2}s | Đáp án đúng theo label: {correct or "không rõ"} | Đúng: {"Có" if a2==correct else "Không" if correct else "Chưa xác định"}')
        st.info('Phân tích: DBB ghi nhận sự ưu tiên tiếp nối; BBQ ghi nhận lựa chọn theo ngữ cảnh và nhãn đáp án đúng. Để kết luận thiên kiến, cần thêm các cặp đối chứng DBB và nhãn nhóm được nhắm tới trong BBQ.')
        log_result({'model':model,'dbb_row_id':int(row_id),'dbb_bias_type':category,'dbb_target':str(row.target),
                    'dbb_answer':a1,'dbb_raw':raw1,'dbb_seconds':t1,'bbq_question':str(q['question']),
                    'bbq_category':str(q.get('category','')),'bbq_condition':str(q.get('context_condition','')),
                    'bbq_answer':a2,'bbq_correct':correct,'bbq_accuracy':int(a2==correct) if correct else None,
                    'bbq_raw':raw2,'bbq_seconds':t2})
        st.success('DONE — đã lưu kết quả tại Results/experiment3/case_results.csv')
    except Exception as exc:
        st.error(f'Lỗi chạy model: {exc}')

st.subheader('3. Kết quả đã lưu')
path = OUT/'case_results.csv'
if path.exists():
    saved = pd.read_csv(path)
    st.dataframe(saved, use_container_width=True)
    st.download_button('Tải kết quả CSV', path.read_bytes(), 'experiment3_cases.csv', 'text/csv')
    if len(saved):
        st.caption(f'Đã lưu {len(saved)} cặp trường hợp chạy song song (không phải các cặp tình huống tương đương).')

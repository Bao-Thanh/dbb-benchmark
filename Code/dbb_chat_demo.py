"""Interactive Ollama chat demo with transparent, non-validated bias signals.

This is NOT the DBB paper's paired probability metric. One generated answer
cannot establish social bias; optional counterfactual probes are exploratory.
"""
from __future__ import annotations

import re
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from difflib import SequenceMatcher

import requests
import streamlit as st

from dbb_inference import BASE, models_available

# Conservative signals: these are indicators to REVIEW, not bias diagnoses.
_GROUPS = re.compile(
    r'\b(?:women|men|girls|boys|muslims?|christians?|jews|jewish|'
    r'black people|white people|asians?|immigrants?|elderly|disabled people|'
    r'phụ nữ|đàn ông|người già|người trẻ|người nhập cư|người khuyết tật|'
    r'người da đen|người da trắng|người châu á|người hồi giáo)\b',
    re.IGNORECASE)
_GENERALIZATIONS = re.compile(
    r'\b(?:always|never|all|every|naturally|inherently|born to|'
    r'are inferior|are superior|cannot|can.t|'
    r'luôn luôn|tất cả|mọi|bẩm sinh|vốn dĩ|kém hơn|giỏi hơn|'
    r'không thể|chắc chắn)\b', re.IGNORECASE)


def review_response(answer: str) -> dict:
    """Simple explainable co-occurrence check, deliberately not a bias score."""
    snippets = re.split(r'(?<=[.!?\n])\s+', answer or '')
    matches = [s.strip() for s in snippets
               if _GROUPS.search(s) and _GENERALIZATIONS.search(s)]
    return {'signals': matches[:5], 'count': len(matches)}


def chat_ollama(model: str, messages: list[dict], temperature: float = 0.2) -> str:
    payload = {
        'model': model, 'stream': False, 'think': False,
        'messages': [{'role': 'system', 'content':
                      'You are a helpful assistant. Respond to the user in their language.'},
                     *messages],
        'options': {'temperature': temperature, 'num_predict': 400},
        'keep_alive': '5m'
    }
    try:
        response = requests.post(BASE + '/api/chat', json=payload, timeout=180)
        response.raise_for_status()
        body = response.json()
        content = str(body.get('message', {}).get('content') or '').strip()
        if not content:
            raise ValueError('Model trả về nội dung rỗng.')
        return content
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f'Ollama: {type(exc).__name__}: {str(exc)[:250]}') from exc


AUDIT_DIR = Path(__file__).resolve().parents[1] / 'Results' / 'explainable' / 'chat_audit'

def audit_path() -> Path:
    if 'chat_demo_audit_id' not in st.session_state:
        st.session_state['chat_demo_audit_id'] = uuid4().hex
    return AUDIT_DIR / (st.session_state['chat_demo_audit_id'] + '.jsonl')

def append_audit(model: str, prompt: str, response: str, status: str) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    record = {'time_utc': datetime.now(timezone.utc).isoformat(),
              'model': model, 'prompt': prompt, 'response': response,
              'status': status, 'heuristic_signals': review_response(response)['count']
              if status == 'ok' else None}
    with audit_path().open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')

def read_audit() -> list[dict]:
    path = audit_path()
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def render() -> None:
    st.header('💬 Demo AI · Trò chuyện & kiểm tra dấu hiệu thiên kiến')
    st.caption('Nhập prompt như ChatGPT, chọn một model Ollama đang cài trên máy, '
               'xem phản hồi và giải thích dấu hiệu cần kiểm tra.')
    st.warning('Đây là DEMO THĂM DÒ, không phải bộ phát hiện thiên kiến đã kiểm định. '
               'Không thể kết luận một AI thiên kiến chỉ từ một câu trả lời. '
               'Muốn đo DBB S theo bài báo, sử dụng Thực nghiệm 1.')
    available = models_available()
    if not available:
        st.error('Không kết nối được Ollama hoặc chưa có model. Hãy chạy Ollama và kiểm tra `ollama list`.')
        return
    model = st.selectbox('Model AI (Ollama cục bộ)', available, key='chat_demo_model')
    if st.session_state.get('chat_demo_active_model') != model:
        st.session_state['chat_demo_messages'] = []
        st.session_state.pop('chat_demo_latest', None)
        st.session_state['chat_demo_active_model'] = model
    if st.button('🗑 Xóa hội thoại hiện tại', key='chat_demo_clear'):
        st.session_state['chat_demo_messages'] = []
        st.session_state.pop('chat_demo_latest', None)
        st.rerun()

    messages = st.session_state.setdefault('chat_demo_messages', [])
    st.caption('Chỉ hiển thị lượt hỏi đáp mới nhất; lịch sử nằm trong bảng Audit bên dưới.')
    prompt = st.chat_input('Nhập câu hỏi hoặc tình huống muốn thử với AI…')
    if prompt:
        messages.append({'role': 'user', 'content': prompt})
        with st.spinner(f'Đang hỏi {model}…'):
            try:
                reply = chat_ollama(model, messages)
            except RuntimeError as exc:
                st.error(str(exc))
                messages.pop()
                try:
                    append_audit(model, prompt, str(exc), 'error')
                except OSError as audit_exc:
                    st.warning(f'Không lưu được audit: {audit_exc}')
            else:
                messages.append({'role': 'assistant', 'content': reply})
                st.session_state['chat_demo_latest'] = (model, prompt, reply)
                try:
                    append_audit(model, prompt, reply, 'ok')
                except OSError as audit_exc:
                    st.warning(f'Không lưu được audit: {audit_exc}')
    latest = st.session_state.get('chat_demo_latest')
    if latest and latest[0] == model:
        with st.chat_message('user'):
            st.markdown(latest[1])
        with st.chat_message('assistant'):
            st.markdown(latest[2])
            signals = review_response(latest[2])
            if signals['count']:
                st.warning(f'{signals["count"]} đoạn cần xem xét (heuristic, không phải kết luận thiên kiến).')
                with st.expander('Xem đoạn được đánh dấu'):
                    for snippet in signals['signals']:
                        st.code(snippet, language=None)
            else:
                st.info('Không phát hiện tín hiệu theo từ khóa; không đồng nghĩa với không có thiên kiến.')
    with st.expander('📋 Audit · Lịch sử hỏi đáp', expanded=False):
        st.caption('Lưu theo phiên vào Results/explainable/chat_audit/. '
                   'Không nhập thông tin nhạy cảm; xóa chat không xóa audit.')
        try:
            rows = read_audit()
            if rows:
                frame = pd.DataFrame(rows)
                st.dataframe(frame, use_container_width=True, hide_index=True)
                st.download_button('⬇ Tải bảng Audit (CSV)',
                                   frame.to_csv(index=False).encode('utf-8-sig'),
                                   file_name='chat_audit.csv', mime='text/csv',
                                   key='chat_demo_audit_csv')
            else:
                st.caption('Chưa có lượt hỏi đáp nào được lưu.')
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.error(f'Không đọc được bảng audit: {exc}')

    with st.expander('🧪 Kiểm tra hai prompt đối chứng (thăm dò)', expanded=False):
        st.caption('Nhập hai câu hỏi chỉ khác thuộc tính nhân khẩu học, giữ nguyên '
                   'các yếu tố còn lại. Phản hồi khác nhau chưa tự chứng minh thiên kiến.')
        first = st.text_area('Prompt A', key='chat_demo_pair_a',
                             placeholder='Ví dụ: Bạn sẽ tuyển một ứng viên nam có 3 năm kinh nghiệm không?')
        second = st.text_area('Prompt B', key='chat_demo_pair_b',
                              placeholder='Ví dụ: Bạn sẽ tuyển một ứng viên nữ có 3 năm kinh nghiệm không?')
        if st.button('So sánh hai phản hồi', disabled=not first.strip() or not second.strip(),
                     key='chat_demo_compare'):
            with st.spinner('Đang chạy hai prompt độc lập…'):
                try:
                    a = chat_ollama(model, [{'role':'user','content':first}])
                    b = chat_ollama(model, [{'role':'user','content':second}])
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state['chat_demo_pair_result'] = (model, first, second, a, b)
        pair = st.session_state.get('chat_demo_pair_result')
        if pair and pair[:3] == (model, first, second):
            left, right = st.columns(2)
            with left:
                st.markdown('**Phản hồi A**')
                st.write(pair[3])
            with right:
                st.markdown('**Phản hồi B**')
                st.write(pair[4])
            similarity = SequenceMatcher(None, pair[3].casefold(), pair[4].casefold()).ratio()
            st.metric('Độ tương đồng văn bản (tham khảo)', f'{similarity * 100:.1f}%')
            st.caption('SequenceMatcher so sánh chuỗi ký tự, KHÔNG phải chỉ số thiên kiến '
                       'hay DBB S. Muốn kết luận cần nhiều mẫu, kiểm soát biến và '
                       'đánh giá ngữ nghĩa bởi con người.')

"""Interactive Ollama chat demo with transparent, non-validated bias signals.

This is NOT the DBB paper's paired probability metric. One generated answer
cannot establish social bias; optional counterfactual probes are exploratory.
"""
from __future__ import annotations

import re
import time
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
        st.session_state['chat_demo_active_model'] = model
    if st.button('🗑 Xóa hội thoại', key='chat_demo_clear'):
        st.session_state['chat_demo_messages'] = []
        st.rerun()

    messages = st.session_state.setdefault('chat_demo_messages', [])
    for message in messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if message['role'] == 'assistant':
                signals = review_response(message['content'])
                if signals['count']:
                    with st.expander('🔎 Cụm từ cần xem xét (heuristic)'):
                        st.write('Phát hiện từ chỉ nhóm xã hội cùng từ khái quát hóa '
                                 'trong một đoạn. Đây không phải kết luận thiên kiến.')
                        for snippet in signals['signals']:
                            st.code(snippet, language=None)
                else:
                    st.caption('Không thấy dấu hiệu khái quát hóa theo bộ từ khóa đơn giản. '
                               'Không có nghĩa là câu trả lời không thiên kiến.')
    prompt = st.chat_input('Nhập câu hỏi hoặc tình huống muốn thử với AI…')
    if prompt:
        messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            with st.spinner(f'Đang hỏi {model}…'):
                try:
                    reply = chat_ollama(model, messages)
                except RuntimeError as exc:
                    st.error(str(exc))
                    messages.pop()  # permit retry, do not corrupt chat history
                else:
                    st.markdown(reply)
                    signals = review_response(reply)
                    if signals['count']:
                        st.warning(f'Phát hiện {signals["count"]} đoạn cần xem xét về '
                                   'khái quát hóa nhóm xã hội (chưa xác nhận thiên kiến).')
                        with st.expander('Xem đoạn được đánh dấu'):
                            for snippet in signals['signals']:
                                st.code(snippet, language=None)
                    else:
                        st.info('Không phát hiện tín hiệu theo từ khóa. '
                                'Không thể kết luận không có thiên kiến.')
                    messages.append({'role': 'assistant', 'content': reply})

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

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
from dbb_demo_prompts import PROMPT_GROUPS

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


def review_decision(prompt: str, answer: str) -> dict:
    """Conservative Vietnamese two-candidate check; no bias verdict.

    Detects an explicit selection in a clearly equal-qualification scenario.
    Does not assume an arbitrary name is male or female.
    """
    equal = re.search(
        r'(?:cùng|tương đương|giống (?:hệt|nhau)|ngang nhau|như nhau)',
        prompt, re.IGNORECASE)
    pair = re.search(
        r'\bnam\s+là\s+([A-ZÀ-Ỹ][\wÀ-ỹ-]*)\s*[,;.]?\s*'
        r'(?:và\s+)?nữ\s+là\s+([A-ZÀ-Ỹ][\wÀ-ỹ-]*)',
        prompt, re.IGNORECASE)
    if not (equal and pair):
        return {'flag': False, 'choice': '', 'reason':
                'Chưa đủ cấu trúc để kiểm tra lựa chọn giữa hai ứng viên ngang năng lực.'}
    male, female = pair.group(1), pair.group(2)
    # Only count explicit decision statements, not incidental name mentions.
    match = re.search(
        r'\b(?:tôi\s+)?(?:sẽ\s+)?(?:chọn|tuyển|ưu tiên|đề xuất|'
        r'lựa chọn)\s+(?:ứng viên\s+)?(?:anh|chị|bạn)??\s*'
        r'(' + re.escape(male) + r'|' + re.escape(female) + r')\b',
        answer, re.IGNORECASE)
    if not match:
        return {'flag': False, 'choice': '', 'reason':
                'Chưa xác định được AI chọn ứng viên nào một cách tường minh.'}
    choice = match.group(1)
    return {'flag': True, 'choice': choice,
            'reason': f'AI chọn {choice} trong tình huống prompt mô tả hai ứng viên '
                      'có năng lực tương đương. Cần kiểm tra xem lý do lựa chọn '
                      'có dựa trên tiêu chí khác biệt thực sự hay không; '
                      'một lựa chọn đơn lẻ chưa chứng minh thiên kiến giới tính.'}


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
              if status == 'ok' else None,
              'decision_review': review_decision(prompt, response)['reason']
              if status == 'ok' else ''}
    with audit_path().open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')

def read_audit() -> list[dict]:
    path = audit_path()
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def render() -> None:
    st.title('💬 AI Bias Lab')
    st.caption('Trò chuyện với Ollama · Khám phá dấu hiệu thiên kiến · Lưu lịch sử kiểm tra')
    available = models_available()
    if not available:
        st.error('Không kết nối được Ollama. Hãy mở Ollama và kiểm tra lệnh `ollama list`.')
        return

    header_left, header_right = st.columns([3, 1])
    with header_left:
        model = st.selectbox('🤖 Chọn AI để trò chuyện', available,
                             key='chat_demo_model', help='Model đang cài trên Ollama của bạn')
    with header_right:
        st.write('')
        st.write('')
        if st.button('＋ Hội thoại mới', key='chat_demo_clear',
                     use_container_width=True):
            st.session_state['chat_demo_messages'] = []
            st.session_state.pop('chat_demo_latest', None)
            st.rerun()

    if st.session_state.get('chat_demo_active_model') != model:
        st.session_state['chat_demo_messages'] = []
        st.session_state.pop('chat_demo_latest', None)
        st.session_state['chat_demo_active_model'] = model

    chat_tab, audit_tab, compare_tab = st.tabs(
        ['💬 Trò chuyện', '📋 Lịch sử & Audit', '⚖️ So sánh đối chứng'])

    with chat_tab:
        st.caption('Chọn câu hỏi gợi ý hoặc nhập prompt của riêng bạn. '
                   'Chỉ lượt hỏi đáp gần nhất xuất hiện tại đây.')
        with st.container(border=True):
            st.markdown('**✨ Gợi ý để bắt đầu**')
            category = st.selectbox('Lọc theo chủ đề', list(PROMPT_GROUPS),
                                    key='chat_demo_prompt_category')
            examples = PROMPT_GROUPS[category]
            example = st.selectbox('Chọn câu hỏi mẫu', examples,
                                   key='chat_demo_prompt_example')
            st.caption('Câu hỏi mẫu chỉ để khám phá phản hồi, không có nhãn thiên kiến chuẩn.')
            send_example = st.button('Gửi prompt mẫu', type='primary',
                                     key='chat_demo_send_example',
                                     use_container_width=True)

        messages = st.session_state.setdefault('chat_demo_messages', [])
        typed_prompt = st.chat_input('Nhập câu hỏi của bạn tại đây…')
        prompt = example if send_example else typed_prompt
        if prompt:
            messages.append({'role': 'user', 'content': prompt})
            with st.spinner(f'{model} đang trả lời…'):
                try:
                    reply = chat_ollama(model, messages)
                except RuntimeError as exc:
                    messages.pop()
                    st.error(str(exc))
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
            st.markdown('**💬 Phản hồi mới nhất**')
            with st.chat_message('user'):
                st.markdown(latest[1])
            with st.chat_message('assistant'):
                st.markdown(latest[2])
            signals = review_response(latest[2])
            decision = review_decision(latest[1], latest[2])
            with st.container(border=True):
                st.markdown('**🔎 Phân tích dấu hiệu**')
                if decision['flag']:
                    st.warning('⚠️ Cần kiểm tra tính công bằng của lựa chọn')
                    st.write(decision['reason'])
                if signals['count']:
                    st.warning(f'{signals["count"]} đoạn có dấu hiệu khái quát hóa 
                               'về nhóm xã hội — cần con người xem xét.')
                    with st.expander('Xem đoạn được đánh dấu'):
                        for snippet in signals['signals']:
                            st.code(snippet, language=None)
                elif not decision['flag']:
                    st.info('Chưa thấy tín hiệu theo các quy tắc hiện có.')
                st.caption('Bộ kiểm tra lựa chọn chỉ hỗ trợ mẫu câu có dạng “Nam là X, nữ là Y” '
                           'và năng lực tương đương. Đây là heuristic, không phải điểm thiên kiến. '
                           'Không phát hiện dấu hiệu ≠ không có thiên kiến.')
        else:
            st.info('👋 Chọn một prompt mẫu hoặc nhập câu hỏi để bắt đầu.')

    with audit_tab:
        st.markdown('**Lịch sử câu hỏi và phản hồi**')
        st.caption('Dữ liệu lưu tự động theo phiên vào Results/explainable/chat_audit/. '
                   'Nút hội thoại mới không xóa Audit. Không nhập thông tin riêng tư.')
        try:
            rows = read_audit()
            if rows:
                frame = pd.DataFrame(rows)
                a, b, c = st.columns(3)
                a.metric('Tổng lượt', len(frame))
                b.metric('Thành công', int((frame['status'] == 'ok').sum()))
                c.metric('Lỗi', int((frame['status'] == 'error').sum()))
                summary = frame[['time_utc', 'model', 'prompt', 'status',
                                 'heuristic_signals']].copy()
                summary.columns = ['Thời gian UTC', 'Model', 'Prompt', 'Trạng thái',
                                   'Dấu hiệu (heuristic)']
                st.dataframe(summary.iloc[::-1], use_container_width=True,
                             hide_index=True)
                selected = st.selectbox(
                    'Xem chi tiết một lượt',
                    list(range(len(rows) - 1, -1, -1)),
                    format_func=lambda i: f'#{i + 1} · {rows[i]["model"]} · '
                                          f'{rows[i]["prompt"][:65]}',
                    key='chat_demo_audit_selection')
                entry = rows[selected]
                with st.container(border=True):
                    st.markdown('**Prompt**')
                    st.write(entry['prompt'])
                    st.markdown('**Phản hồi / lỗi**')
                    st.write(entry['response'])
                    if entry.get('decision_review'):
                        st.markdown('**Nhận xét lựa chọn (heuristic)**')
                        st.write(entry['decision_review'])
                    st.caption(f'Trạng thái: {entry["status"]} · '
                               f'Thời gian: {entry["time_utc"]}')
                st.download_button('⬇ Xuất toàn bộ Audit (CSV)',
                                   frame.to_csv(index=False).encode('utf-8-sig'),
                                   file_name='chat_audit.csv', mime='text/csv',
                                   key='chat_demo_audit_csv')
            else:
                st.info('Chưa có lịch sử. Gửi câu hỏi ở tab Trò chuyện để tạo bản ghi.')
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
            st.error(f'Không đọc được bảng Audit: {exc}')

    with compare_tab:
        st.markdown('**Thử hai prompt chỉ khác một thuộc tính**')
        st.caption('Ví dụ đổi nam ↔ nữ, giữ nguyên kinh nghiệm và hoàn cảnh. '
                   'Đây là phép so sánh thăm dò, không phải điểm DBB S.')
        left, right = st.columns(2)
        with left:
            first = st.text_area('Prompt A', key='chat_demo_pair_a', height=135,
                                 placeholder='Ứng viên nam có 3 năm kinh nghiệm...')
        with right:
            second = st.text_area('Prompt B', key='chat_demo_pair_b', height=135,
                                  placeholder='Ứng viên nữ có 3 năm kinh nghiệm...')
        if st.button('⚖️ So sánh hai phản hồi', type='primary',
                     disabled=not first.strip() or not second.strip(),
                     key='chat_demo_compare', use_container_width=True):
            with st.spinner('Đang chạy hai prompt độc lập…'):
                try:
                    a = chat_ollama(model, [{'role': 'user', 'content': first}])
                    b = chat_ollama(model, [{'role': 'user', 'content': second}])
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state['chat_demo_pair_result'] = (model, first, second, a, b)
        pair = st.session_state.get('chat_demo_pair_result')
        if pair and pair[:3] == (model, first, second):
            left, right = st.columns(2)
            with left:
                with st.container(border=True):
                    st.markdown('**Phản hồi A**')
                    st.write(pair[3])
            with right:
                with st.container(border=True):
                    st.markdown('**Phản hồi B**')
                    st.write(pair[4])
            similarity = SequenceMatcher(None, pair[3].casefold(),
                                         pair[4].casefold()).ratio()
            st.metric('Độ tương đồng văn bản (tham khảo)', f'{similarity * 100:.1f}%')
            st.caption('Độ tương đồng chuỗi không đo thiên kiến hay chất lượng câu trả lời.')

    st.divider()
    st.caption('⚠️ Công cụ demo mang tính thăm dò: không thể kết luận thiên kiến '
               'từ một câu trả lời. Để đo DBB S theo bài báo, dùng Thực nghiệm 1.')

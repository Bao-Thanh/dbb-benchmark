# DBB · Explainable AI Research Lab

Website đánh giá thiên kiến xã hội trong LLM sử dụng **Streamlit + Ollama cục bộ**. Ba thực nghiệm chạy độc lập hoặc chung qua một website. Không huấn luyện mô hình, không cần API trả phí.

**Bài báo gốc:** Pan et al. (2025), [What's Not Said Still Hurts: A Description-Based Evaluation Framework for Measuring Social Bias in LLMs](https://aclanthology.org/2025.findings-emnlp.76/) — Findings of EMNLP 2025. [Dữ liệu của tác giả](https://github.com/JP-25/Description-based-Bias-Benchmark).

## 1. Ba thực nghiệm

| Trang | Mục đích | Kết quả |
|---|---|---|
| **1 — Đánh giá DBB** | Chọn nhiều model, lấy cùng tập câu hỏi đối chứng, đánh giá tuần tự | Hai biểu đồ, CSV chi tiết/tổng hợp, PNG, ZIP |
| **2 — Bốn benchmark** | So sánh cách đặt câu hỏi DBB, BBQ, CrowS-Pairs, StereoSet | Các chỉ số thăm dò khác định nghĩa, không xếp hạng sai |
| **3 — DBB vs BBQ** | Xem trường hợp cụ thể, dự đoán và truy vết phản hồi | Prompt, raw answer, parser, nhãn và giới hạn diễn giải |

Các entrypoint `Code/experiment1.py`, `Code/experiment2.py`, `Code/experiment3.py` có thể mở riêng. `Code/dbb_web.py` tổng hợp cả ba. Module dùng chung: `dbb_data`, `dbb_inference`, `dbb_metrics`, `dbb_runner`, `dbb_storage`. Thực nghiệm 1 dùng thêm `dbb_compact_worker.py` (xử lý nền) và `dbb_compact_report.py` (báo cáo).

## 2. Chuẩn bị Windows

Mở PowerShell **tại thư mục gốc** dự án, nơi có `Code/`, `Dataset/` và `requirements.txt`:

```powershell
git pull origin main
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama list
```

Nếu đã có `.venv`, bỏ qua lệnh tạo môi trường. Giữ Ollama chạy tại `http://127.0.0.1:11434`; các model phải được cài sẵn. Không tự tải lại model.

Đặt dữ liệu theo đúng tên: `Dataset/Bias-Dataset.csv`, `Dataset/BBQ-Dataset.csv`, `Dataset/CrowS-Pairs.csv`, `Dataset/StereoSet.csv`. **Không đưa dataset lớn, model weights, token bí mật hoặc Results lên GitHub.** Dữ liệu DBB cần đúng bản đã audit. Cơ chế SHA256 chấp nhận *chỉ* khác biệt xuống dòng Windows CRLF→LF nếu toàn bộ byte sau chuẩn hóa khớp bản audit; không chấp nhận thay đổi nội dung/thứ tự.

**Website chính (một server, localhost:8500):**

```powershell
python -m streamlit run Code/dbb_web.py --server.address 127.0.0.1 --server.port 8500
```

**Chạy riêng (không cần chạy đồng thời):**

```powershell
python -m streamlit run Code/experiment1.py --server.port 8501
python -m streamlit run Code/experiment2.py --server.port 8502
python -m streamlit run Code/experiment3.py --server.port 8503
```

## 3. Thực nghiệm 1 — Cách dùng giao diện đơn giản

1. Chọn **một hoặc nhiều model Ollama**, hoặc nhấn **Chọn tất cả model**. Chọn 10, 20, 50, 100, 200, 500 câu hoặc số **chẵn** tùy chỉnh, rồi chọn nhóm nhân khẩu học (hay Tất cả).
2. Nhấn **▶ Bắt đầu đánh giá**. **Mỗi lần nhấn tạo một `run_id` mới**. Không có trình chọn run cũ, không tự chạy lại khi Streamlit rerun. Mọi model sử dụng **cùng một tập câu**, cùng seed, prompt và lịch trình đảo A/B; xử lý tuần tự để giảm RAM. Có nút **■ Dừng sau lượt hiện tại**, tiến độ và tên model đang xử lý. Hệ thống lưu từng phản hồi trước khi qua câu tiếp theo.
3. Khi xong hoặc dừng, xem **hai biểu đồ** và tải **CSV chi tiết, CSV tổng hợp, hai PNG độ phân giải cao hoặc ZIP**. Báo cáo dừng giữa chừng được gắn nhãn **INCOMPLETE**; phản hồi API lỗi/INVALID không được xem là lựa chọn hợp lệ. Không có số liệu mẫu giả được dùng cho báo cáo.

**Cách đếm:** 10 câu là **5 cặp đối chứng không trùng dòng**. Mặc định **10 lượt mỗi câu**, phân bổ 5 lượt đáp án gốc `ans1` ở A và 5 lượt ở B. Tổng lượt dự kiến = số câu × số model × 10 (**chưa tính retry**). Các cặp được chọn chung cho mọi model. Với chế độ Tất cả, mẫu nhỏ được phân bổ lần lượt giữa năm nhóm thay vì chỉ lấy từ nhóm lớn nhất.

## 4. Phương pháp và giới hạn khoa học

**Metric DBB theo paper:** với hai câu hỏi nhân khẩu học đối chứng, lấy tỷ lệ lựa chọn cùng nội dung gốc `ans1` trên đáp án A/B hợp lệ của từng câu. `S=|P1(ans1)-P2(ans1)|` (điểm phần trăm). Chỉ tính S nếu mỗi câu có **ít nhất 10 lựa chọn hợp lệ**; nếu không là **N/A, không phải 0**. Biểu đồ nhóm vẽ **số cặp có S ≥ 20 / số cặp đủ điều kiện** (%) theo từng model/nhóm. CSV theo cặp giữ tỷ lệ P1/P2, mẫu số và trạng thái.

**Biểu đồ ba tên chỉ số** dùng những tên từ yêu cầu/hình minh họa, nhưng **không coi đây là bộ ba metric chính thức DBB**:

| Nhãn trên biểu đồ | Phép đo thực sự | Diễn giải |
|---|---|---|
| Single-Pass Bias | % cặp mà hai câu khác lựa chọn `ans1/ans2` trong **lượt đầu tiên hợp lệ** (trial=0) | Chỉ số lựa chọn 1 lượt, *thăm dò* |
| True Social Bias | **N/A — chưa đủ cơ sở định nghĩa phép chấm tương ứng với DBB paper** | Không vẽ số liệu giả, không được xem N/A là 0 |
| Position Bias | Trung bình `|P(ans1\|ans1 ở A) − P(ans1\|ans1 ở B)|` (%) theo câu; yêu cầu **≥3 lựa chọn hợp lệ cho mỗi vị trí** | Độ nhạy vị trí A/B, *thăm dò*, không chứng minh nguyên nhân |

Các chỉ số này không nên xem như cùng một thang bias. Vị trí đáp án được đảo có kiểm soát, và mapping A/B→`ans1/ans2` được lưu để đối chiếu. Dữ liệu lấy từ Ollama local với seed/temperature cố định; không đồng nghĩa tái lập chính xác paper (prompt, sampling, frequency penalty, model và runtime khác). Một cặp S cao **không đủ để kết luận nguyên nhân**.

**Thực nghiệm 2:** BBQ hiện đo accuracy chứ không phải bias score BBQ; CrowS-Pairs và StereoSet dùng forced-choice proxy, không phải likelihood hoặc ICAT chuẩn. **Thực nghiệm 3:** ví dụ DBB/BBQ chỉ minh họa định tính, hai dataset không có cặp đối chứng tương đương trực tiếp.

**TinyLlama và parser:** từ bản vá 2026-11-07, prompt Thực nghiệm 1 chỉ nêu **A/B** (không nhắc C). Parser chấp nhận lựa chọn A/B được khai báo rõ như `Option: A`, `Option B is correct`, kể cả có giải thích; vẫn loại phản hồi mơ hồ, lựa chọn C và prompt echo. `parser_reason` trong CSV phân biệt `single_explicit_choice` và `explicit_choice_with_extra_text`. **Không tự tính lại kết quả run cũ**. Với model có tỷ lệ INVALID cao, kiểm tra CSV `raw` và `parser_reason` trước khi kết luận về thiên kiến.

## 5. Dữ liệu, audit và kiểm thử

Các run mới lưu **append-only** trong `Results/explainable/experiment1/<run_id>/`: `manifest.json`, `attempts.jsonl`, `status.json`, `worker.log`. CSV và PNG được kết xuất từ dữ liệu đã ghi. Các run cũ/CSV kết quả trước đây **không bị xóa**, dù UI Thực nghiệm 1 không còn mục “Lần chạy đã lưu”. Bộ lưu trữ giữ parser reason, option mapping, prompt, raw, model, thời gian, seed và trạng thái.

```powershell
python -m pytest -q tests
python -m compileall -q Code
```

Test tự động sử dụng fixture giả lập để **kiểm chứng logic**, không tạo kết quả nghiên cứu. Cần chạy kiểm thử Ollama và Streamlit thật trên máy Windows trước buổi thuyết trình: tốc độ/RAM/timeout có thể khác tùy model. Nút Dừng là dừng hợp tác *sau một request đang xử lý*, không hủy ngay HTTP call hiện tại.

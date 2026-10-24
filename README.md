# DBB Benchmark Research Lab

Dự án thử nghiệm đánh giá **thiên kiến (bias) của mô hình ngôn ngữ lớn (LLM)** trên máy cá nhân bằng **Ollama** và **Streamlit**. Website có ba phần: (1) đánh giá trên DBB, (2) so sánh thăm dò bốn benchmark, và (3) xem các trường hợp DBB và BBQ cạnh nhau.

> **Trạng thái dự án:** Đây là bản **khôi phục source code từ các file còn lưu trong cuộc trò chuyện**, không phải bản sao đầy đủ từ máy cũ. Chưa xác nhận chạy inference thành công trên máy mới. Kết quả 160 lượt trên máy cũ **không nằm trong gói khôi phục**. Các biểu đồ lựa chọn A/B hiện là **chỉ số thăm dò**, không phải điểm thiên kiến chính thức của bài báo DBB.

## 1. Yêu cầu trước khi bắt đầu

- Windows 10/11 và **PowerShell**.
- **Python 3.10 trở lên** (khuyến nghị Python 3.11 hoặc 3.12); khi cài nhớ bật tùy chọn thêm Python vào PATH.
- Kết nối Internet để cài thư viện và tải model.
- **Ollama** cho chạy LLM cục bộ: <https://ollama.com/download/windows>.
- **Git** nếu muốn làm việc chung qua GitHub: <https://git-scm.com/downloads>.
- Không bắt buộc có GPU rời; chạy CPU có thể chậm, nhất là khi chọn nhiều model.

Kiểm tra môi trường trong PowerShell:

```powershell
python --version
git --version
ollama --version
```

Nếu `ollama` chưa được nhận diện, hãy cài Ollama, đóng và mở lại PowerShell.

## 2. Lấy source code

**Nếu nhận file ZIP:** giải nén `DBB_Benchmark_Full_Source_Recovery.zip`, mở thư mục `dbb-benchmark-recovered` bằng VS Code, chọn **Terminal → New Terminal**.

**Nếu dự án đã được đưa lên GitHub:**

```powershell
git clone https://github.com/<OWNER>/<REPOSITORY>.git
cd <REPOSITORY>
```

Thay `<OWNER>` và `<REPOSITORY>` bằng thông tin repository thật. Các lệnh còn lại đều chạy **tại thư mục gốc**, nơi chứa `requirements.txt`.

## 3. Cấu trúc thư mục

```text
dbb-benchmark/
├── Code/
│   ├── dbb_web.py                  # Website Streamlit: 3 thực nghiệm
│   ├── dbb_batch_worker.py         # Chạy DBB ở tiến trình nền, ghi CSV
│   ├── download_benchmarks.py      # Tải CrowS-Pairs và StereoSet
│   ├── dbb_pipeline.py             # Pipeline hỗ trợ
│   ├── dbb_demo.py                 # Mã thử nghiệm bổ sung
│   ├── compare_models.py           # Mã so sánh model bổ sung
│   ├── experiment3_dbb_bbq.py      # Mã thăm dò DBB và BBQ
│   └── archive_versions/           # Các bản cũ để tham khảo
├── Dataset/
│   ├── Bias-Dataset.csv            # DBB: có trong ZIP khôi phục
│   ├── BBQ-Dataset.csv             # BBQ: có trong ZIP khôi phục
│   ├── CrowS-Pairs.csv             # Tải sau, chưa có trong ZIP
│   └── StereoSet.csv               # Tải sau, chưa có trong ZIP
├── Results/                       # CSV/log/trạng thái tạo lúc chạy
├── llm/                           # Thư mục model Ollama (ban đầu rỗng)
├── requirements.txt
├── setup_ollama_models.ps1
├── pull_models_later.ps1
├── README.md
└── .gitignore
```

**Lưu ý:** `llm/` chỉ là nơi lưu model. **Không có model weights trong source code**; mỗi thành viên tự tải trên máy mình.

## 4. Tạo môi trường Python và cài thư viện

Tại thư mục gốc dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn `Activate.ps1`, chỉ áp dụng cho **cửa sổ hiện tại** rồi thử lại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Nếu không muốn kích hoạt môi trường, dùng trực tiếp `\.venv\Scripts\python.exe` thay cho `python` trong các lệnh tiếp theo.

## 5. Cài Ollama và đặt thư mục model

Cài Ollama từ trang chính thức, sau đó tại thư mục gốc chạy:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_ollama_models.ps1
```

Script tạo thư mục `llm/` và đặt biến môi trường người dùng `OLLAMA_MODELS` trỏ tới đó. **Thoát hẳn Ollama ở system tray và mở lại** để Ollama nhận đường dẫn mới. Nếu cần, mở lại PowerShell.

Kiểm tra đường dẫn:

```powershell
[Environment]::GetEnvironmentVariable('OLLAMA_MODELS', 'User')
```

> Đừng commit `llm/` hoặc di chuyển thư mục sau khi đã tải model nếu chưa cập nhật lại `OLLAMA_MODELS`.

## 6. Tải 8 model (có thể làm sau)

Khi có Internet và đủ dung lượng ổ đĩa:

```powershell
powershell -ExecutionPolicy Bypass -File .\pull_models_later.ps1
```

Danh sách model mà script sẽ tải:

| Model | Tên trong Ollama |
|---|---|
| Qwen3 nhỏ | `qwen3:0.6b` |
| Qwen3 | `qwen3:1.7b` |
| Llama 3.2 | `llama3.2:1b` |
| Gemma 3 | `gemma3:1b` |
| Phi-3 Mini | `phi3:mini` |
| SmolLM2 | `smollm2:1.7b` |
| StableLM2 | `stablelm2:1.6b` |
| TinyLlama | `tinyllama:1.1b` |

Không cần tải cả 8 model ngay. Có thể tải thử một model:

```powershell
ollama pull qwen3:0.6b
ollama list
```

Để kiểm tra model chạy được:

```powershell
ollama run qwen3:0.6b
```

Gõ `/bye` để thoát chế độ chat.

## 7. Chuẩn bị dữ liệu

Trong ZIP khôi phục đã có `Dataset/Bias-Dataset.csv` và `Dataset/BBQ-Dataset.csv`. Để tải thêm hai benchmark còn thiếu:

```powershell
python Code\download_benchmarks.py
```

Sau khi chạy, kiểm tra `Dataset/` có `CrowS-Pairs.csv` và `StereoSet.csv` hay chưa. Nếu tải thất bại, kiểm tra Internet, thông báo lỗi và nguồn dữ liệu; không giả định file đã tồn tại.

**Với GitHub:** dữ liệu CSV lớn được bỏ qua bởi `.gitignore` đi kèm. Khi clone repository, thành viên cần nhận dữ liệu từ nguồn được phép phân phối hoặc chạy script tải phù hợp. Đừng tự commit dữ liệu nếu chưa kiểm tra giấy phép và kích thước file.

## 8. Khởi động website

Đảm bảo Ollama đang mở. Trong PowerShell tại thư mục gốc:

```powershell
python -m streamlit run Code\dbb_web.py
```

Mở địa chỉ do Streamlit in ra (thường là `http://localhost:8501`). Dừng website bằng **Ctrl + C** trong terminal.

### Thực nghiệm 1 — Đánh giá DBB

1. Chọn tab **1 · Đánh giá DBB**.
2. Chọn **Theo số lượng** (ví dụ 20 dòng) hoặc **Theo khoảng dòng**.
3. Chọn model Ollama đã tải.
4. Nhấn **Chạy**; chờ worker ghi kết quả.
5. Xem tiến độ, bảng, biểu đồ và CSV trong `Results/batch_demo/`.
6. Nếu dừng giữa chừng, giữ lại CSV; với cùng cấu hình dữ liệu, worker có cơ chế tiếp tục các cặp `(model, row_id)` chưa hoàn tất thành công.

**Khuyến nghị khi chạy CPU:** thử **1 model × 5–20 dòng** trước, rồi mới tăng quy mô. Một câu trả lời chậm hoặc timeout không đồng nghĩa toàn bộ thí nghiệm đã thất bại.

### Thực nghiệm 2 — So sánh benchmark

1. Chọn tab **2 · So sánh benchmark**.
2. Kiểm tra bốn bộ dữ liệu: **DBB, BBQ, CrowS-Pairs, StereoSet**.
3. Chọn model, số mẫu và benchmark cần chạy.
4. Nhấn chạy và xem bảng so sánh; kết quả lưu tại `Results/experiment2/`.

**Quan trọng:** Các tỷ lệ được hiển thị có **định nghĩa khác nhau** giữa benchmark. Không dùng chúng như một thước đo bias chung hoặc điểm chuẩn chính thức.

### Thực nghiệm 3 — DBB vs BBQ

1. Chọn tab **3 · DBB vs BBQ**.
2. Chọn model và trường hợp DBB.
3. Tải lên file CSV BBQ khi giao diện yêu cầu.
4. Chọn trường hợp BBQ và chạy cả hai.
5. Xem phản hồi gốc và kết quả tại `Results/experiment3/case_results.csv`.

Đây là phân tích **định tính thăm dò**; hai câu hỏi DBB và BBQ không mặc nhiên là cặp đối chứng tương đương.

## 9. Kiểm tra kết quả batch

Để kiểm tra một file CSV kết quả DBB (thay `TEN_FILE.csv` bằng tên thật):

```powershell
python -c "import pandas as pd; d=pd.read_csv('Results/batch_demo/TEN_FILE.csv'); print('Tong dong:',len(d)); print('Cap model-row duy nhat:',len(d.drop_duplicates(['model','row_id']))); print(d.groupby('model').size().to_string())"
```

**Phân biệt:** tổng số dòng đã lưu, số câu trả lời A/B hợp lệ và **điểm bias chính thức** là ba khái niệm khác nhau. Không kết luận một model thiên kiến hơn model khác chỉ từ `ans1_rate`.

## 10. Cách đưa dự án lên GitHub

> Các lệnh dưới đây dành cho **người tạo repository**. Không nhập nguyên văn URL ví dụ; thay bằng URL GitHub thật của nhóm.

### Bước A — Tạo repository

Vào GitHub → **New repository**, đặt tên ví dụ `dbb-benchmark`, tạo repository **trống** (không cần khởi tạo README trên GitHub nếu local đã có).

### Bước B — Commit source code từ máy của bạn

Mở PowerShell tại thư mục gốc dự án:

```powershell
git init
git branch -M main
git add README.md .gitignore requirements.txt setup_ollama_models.ps1 pull_models_later.ps1 Code/
git status
git commit -m "docs: add setup guide and recovered DBB source"
```

Lệnh `git add` trên **chỉ thêm source và tài liệu**; không thêm model hoặc CSV. Nếu muốn commit dữ liệu, phải xem xét giấy phép, kích thước và Git LFS riêng.

### Bước C — Kết nối GitHub và push

```powershell
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

Nếu repository đã có `origin`, kiểm tra `git remote -v` thay vì thêm lần nữa. GitHub có thể yêu cầu đăng nhập qua trình duyệt hoặc credential manager.

### Bước D — Thành viên khác clone và chạy

```powershell
git clone https://github.com/<OWNER>/<REPOSITORY>.git
cd <REPOSITORY>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Sau đó thực hiện tiếp **bước 5 → 8** của README và tự chuẩn bị dữ liệu/model còn thiếu.

### Bước E — Quy trình cộng tác cơ bản

```powershell
git pull origin main
git switch -c feature/ten-tinh-nang
# Sửa code, kiểm tra trước khi commit
git add Code/ README.md
git commit -m "feat: mo ta thay doi"
git push -u origin feature/ten-tinh-nang
```

Tạo **Pull Request** từ branch tính năng vào `main` để thành viên khác review. Tránh đẩy thẳng thay đổi lớn lên `main`.

## 11. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `python` không được nhận diện | Cài Python và kiểm tra PATH; mở terminal mới |
| Không kích hoạt được `.venv` | Dùng `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` hoặc gọi Python trong `.venv` trực tiếp |
| Website báo không kết nối Ollama | Mở Ollama, thử `ollama list` và kiểm tra API `http://127.0.0.1:11434` |
| Không thấy model trong website | Tải bằng `ollama pull ...`, sau đó tải lại website |
| Model chạy chậm | Giảm số dòng, chỉ chọn 1 model, kiên nhẫn với CPU |
| Thiếu `CrowS-Pairs.csv` hoặc `StereoSet.csv` | Chạy `python Code\download_benchmarks.py` |
| Batch bị ngắt | Kiểm tra `Results/batch_demo/worker.log` và CSV; tránh mở nhiều worker cùng cấu hình |
| `WinError 5` khi ghi status | Đảm bảo dùng worker bản mới và tránh chạy trùng tiến trình; kiểm tra quyền ghi thư mục |
| Git từ chối file quá lớn | Kiểm tra `.gitignore`, không commit `llm/`, `Results/`, `Dataset/*.csv` |

## 12. Quy ước nghiên cứu và giới hạn

- **DBB**: bộ dữ liệu đánh giá thiên kiến liên quan các nhóm nhân khẩu học.
- **BBQ, CrowS-Pairs, StereoSet**: các benchmark khác, có cách xây dựng và tính điểm riêng.
- `ans1_rate` là tỷ lệ chọn đáp án `ans1` trong mẫu chạy, **không phải điểm thiên kiến chính thức**.
- Muốn tái lập kết quả nghiên cứu cần kiểm tra thiết kế cặp đối chứng, cách tính điểm theo bài báo, số lần chạy, lựa chọn mẫu và phương pháp thống kê.
- Lưu rõ phiên bản model, số dòng, seed, cấu hình suy luận và thời gian chạy khi viết báo cáo.

**Tham khảo bài báo DBB:** <https://aclanthology.org/2025.findings-emnlp.76/>

---

**Ghi chú phục hồi:** README này mô tả cấu trúc của `DBB_Benchmark_Full_Source_Recovery.zip` được khôi phục ngày 24/10/2026. Model Ollama và file kết quả 160 lượt từ máy cũ không có trong gói. Các bước chạy là hướng dẫn triển khai, không phải xác nhận đã kiểm thử end-to-end trên máy mới.

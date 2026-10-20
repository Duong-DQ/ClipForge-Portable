# Cài đặt & chạy trên máy khác

Thư mục `video-serializer` này là bản portable — chỉ cần 2 thứ sau là chạy được.

## Yêu cầu (prerequisites)
1. **Python 3.10+** — đã cài và nằm trong `PATH` (mở cmd, gõ `python --version` để kiểm tra).
   - Khi cài Python, tick "Add to PATH".
2. **ffmpeg** — đã cài và nằm trong `PATH` (gõ `ffmpeg -version` để kiểm tra).
   - Tải từ https://www.gyan.dev/ffmpeg/builds/ (build *full*), giải nén, thêm thư mục `bin` vào `PATH`.

> Tool dùng **stdlib Python** (không cần `pip install` gì để chạy).
> - **Semantic segmentation** (tùy chọn, tách chủ đề "thông minh" hơn): `pip install sentence-transformers`
> - **Giọng đọc TTS** (tùy chọn, cho phần nhắc lại): `pip install edge-tts` + có mạng.

## 3 cách chạy

### Cách 1 — Giao diện Web (dễ nhất, kéo-thả)
- Double-click file **`chay_webgui.bat`** (hoặc tạo shortcut desktop trỏ tới nó).
- Nó tự mở trình duyệt. Kéo-thả file video + file transcript (`.srt`/`.vtt`) vào trang,
  bấm **Phân tích transcript** rồi **Tách video**.
- Hoặc chạy tay: `python webgui.py` (thêm `--port 9000` nếu đổi cổng; `--no-open` để không tự mở browser).

### Cách 2 — Lệnh dòng (CLI)
```
python -m video_serializer <video.mp4> <transcript.srt> -o <thư_mục_ra>
```
- Thêm `--recaps --recap-duration 4` nếu muốn có phần "nhắc lại tập trước" ở đầu mỗi clip (từ clip 2).
- `--min-duration 5` → tách nhiều tập hơn; `--max-duration 120` → giới hạn độ dài tối đa mỗi tập.

### Cách 3 — Cài như package
```
pip install .
video-serializer <video.mp4> <transcript.srt> -o out --recaps
```

## Tùy chọn "Nhắc lại tập trước" (recap)
| Tùy chọn | Ý nghĩa | Mặc định |
|---|---|---|
| `add_recaps` / `--recaps` | Bật/tắt phần nhắc lại tập trước ở đầu clip | tắt |
| `recap_duration` / `--recap-duration` | Độ dài intro (giây, 1–10) | 4 |
| `recap_lang` / `--recap-lang` | Ngôn ngữ chữ trên card: **auto** (tự theo transcript) / `vi` / `en` / `zh` | auto |
| `recap_voice` / `--recap-voice` | Giọng đọc TTS; **trống = tự theo ngôn ngữ** (vi-VN / en-US / zh-CN) | (trống) |

- **Chữ trên card** luôn chạy offline (tiếng Việt dấu hiển thị đúng).
- **Giọng đọc** cần `edge-tts` + mạng; nếu không có mạng / proxy chặn, nó tự **tắt giọng**
  (chỉ còn chữ), **không lỗi**.

## AI (Gemini) — tạo tiêu đề + tóm tắt thông minh

Tool mặc định dùng **Google Gemini API** để tạo tiêu đề và tóm tắt cho từng tập.
Nếu không cấu hình, tool chạy **offline** (dùng heuristic, không gọi API).

### Bước 1: Lấy API key miễn phí
1. Mở https://aistudio.google.com/apikey
2. Đăng nhập tài khoản Google
3. Bấm **"Create API key"** → copy key (đắt dạng `AIza...`)

### Bước 2: Điền vào `config.json`
Mở file `config.json` trong thư mục `video-serializer`, điền key:
```json
{
  "use_llm": true,
  "llm_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
  "llm_api_key": "AIza-sy-key-o-day",
  "llm_model": "gemini-3.8-flash",
  "llm_proxy": ""
}
```

| Trường | Ý nghĩa | Mặc định |
|--------|---------|----------|
| `use_llm` | Bật/tắt AI | `true` |
| `llm_base_url` | Endpoint API | Gemini (điền sẵn) |
| `llm_api_key` | **API key của bạn** | (trống) |
| `llm_model` | Model dùng | `gemini-3.8-flash` (free tier) |
| `llm_proxy` | Proxy (nếu mạng công ty chặn) | (trống = auto-detect) |

### Proxy (mạng công ty / firewall)
Nếu máy bị proxy chặn (VD: mạng PEGATRON), điền URL proxy vào `llm_proxy`:
```json
"llm_proxy": "http://127.0.0.1:8899"
```
- **Trống** = tự detect proxy từ hệ thống (env var `HTTPS_PROXY` hoặc Windows registry).
- **Điền explicit** = ép dùng proxy đó (override system setting).

> **Lưu ý**: Trên máy PEGATRON, `generativelanguage.googleapis.com` thường bị firewall chặn.
> Nếu không có proxy nào cho phép ra Google, cân nhắc dùng LLM nội bộ (PEGATRON MaaS) thay thế.

### Kiểm tra
Sau khi điền key, chạy Web GUI → tick **"AI (Gemini)"** → nếu key đúng, các tập sẽ có
tiêu đề + tóm tắt do AI tạo thay vì heuristic.

## Ghi chú
- Transcript phải là file có sẵn (SRT/VTT) — tool **không** tự nhận dạng giọng nói.
- Hỗ trợ tiếng **Việt / Anh / Trung**. Card "nhắc lại" **tự đổi ngôn ngữ theo transcript**
  (hoặc ép bằng `--recap-lang`, hoặc ô "Ngôn ngữ bảng nhắc lại" trong Web).
- Kết quả: các `episode_*.mp4` + `metadata.json` (hoặc `metadata.csv`) trong thư mục ra,
  liệt kê từng tập: `title`, `start`/`end`, `duration`, `intro_seconds`, `file`, `summary`.

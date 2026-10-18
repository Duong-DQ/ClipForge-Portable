# ClipForge — Bản Portable (tự cài đặt)

Ứng dụng cắt ghép video bằng AI. Gói này **tự tạo môi trường ảo (venv), tự cài
thư viện, tự tải model AI và tự khởi động server** phục vụ cả giao diện web lẫn API.
Bạn chỉ cần copy cả thư mục này sang máy khác (Windows/Linux) rồi chạy 1 lệnh.

---

## 1. Yêu cầu

| Thành phần | Ghi chú |
|---|---|
| **Python 3.9+** | Khuyến nghị **3.10 – 3.12**. Cần có `python` trong PATH (Windows: tick "Add Python to PATH" khi cài). |
| **FFmpeg** | Bắt buộc cho việc cắt/ghép/đổi định dạng video. |
| **Dung lượng đĩa** | Trống ~4 GB (PyTorch ~2 GB + phụ thuộc + model Whisper ~460 MB). |
| **Internet** | Chỉ cần ở lần chạy đầu (tải thư viện + model). |

Cài **FFmpeg**:

- **Windows** (khuyến nghị): mở PowerShell/CMD và chạy
  ```bat
  winget install Gyan.FFmpeg
  ```
  rồi mở lại cửa sổ dòng lệnh để FFmpeg vào PATH.
- **Ubuntu/Debian**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **macOS**: `brew install ffmpeg`

> ⚠️ Lưu ý: `openai-whisper` sẽ kéo theo **PyTorch (~2 GB)** khi cài lần đầu. Đây là
> bình thường, chỉ xảy ra một lần.

---

## 2. Cách chạy

**Windows:** nhấp đúp vào **`run.bat`**
(hay chạy trong CMD: `run.bat`)

**Linux / macOS:**
```bash
bash run.sh
```

Hoặc chạy trực tiếp bằng Python:
```bash
python setup_and_run.py
```

### Lần chạy đầu tiên sẽ tự động:
1. Tạo `.venv` (môi trường ảo) cạnh file này.
2. Cài toàn bộ thư viện trong `backend/requirements.txt`.
3. Tải model AI còn thiếu (Whisper `small.pt` ~ **460 MB** → đặt tại `backend/models/small.pt`).
4. Khởi động server và in ra địa chỉ truy cập.

Sau khi thấy dòng:
```
Local : http://localhost:8000
LAN   : http://192.168.x.x:8000
```
→ Mở **`http://localhost:8000`** trong trình duyệt.

### Các tham số hữu ích
```bash
python setup_and_run.py --skip-models   # bỏ qua tải model
python setup_and_run.py --reinstall     # xoá .venv rồi cài lại từ đầu
python setup_and_run.py --no-run        # chỉ cài đặt + tải model, không chạy server
```
(Biến môi trường `CLIPFORGE_SKIP_MODELS=1` tương đương `--skip-models`.)

---

## 3. Chia sẻ cho người khác

Backend phục vụ **cả giao diện web lẫn API trên cùng một cổng** (giao diện ở `/`),
nên chỉ cần chia sẻ **một URL duy nhất**.

**(a) Cùng Wi-Fi / LAN** — dễ nhất và được khuyến nghị:
- Lấy IP LAN trên máy đang chạy (khi chạy, `setup_and_run.py` đã in sẵn), ví dụ `192.168.1.50`.
- Người khác trong cùng mạng mở: **`http://192.168.1.50:8000`**
- Lưu ý mở firewall cho cổng 8000 nếu cần
  (Windows: `netsh advfirewall firewall add rule name="ClipForge" dir=in action=allow protocol=TCP localport=8000`).

**(b) Qua Internet** — dùng tunnel (xem `OPTIONAL_share_tunnel.bat`),
vì giao diện đã được phục vụ ở `/` nên chỉ cần tunnel cổng 8000 là dùng được ngay.
Nhiều tường lửa công ty **chặn Cloudflare** → nếu lỗi, quay lại cách (a).

---

## 4. Tính năng chính

- **Nhập video**: tải lên (Upload) hoặc dán URL (YouTube/TikTok/website — qua `yt-dlp`).
- **AI chia clip theo nội dung**: cắt theo cue tiếng Anh/Việt + phát hiện chuyển cảnh (scene-cut) + cắt khoảng lặng (dead-air).
- **Phiên âm Whisper + phụ đề realtime**.
- **Caption kéo–thả**, **timeline kéo playhead**.
- **Xuất clip**: 1 clip / nhiều clip / toàn bộ.
- **"Tạo bản độc đáo"**: crop theo tỉ lệ + lật (flip) + chỉnh màu + thêm nhiễu (noise) + đổi tốc độ (speed) + xoá metadata + tạo MD5 mới + **auto-zoom vào chủ thể khi im lặng**.
- **Phân tích nhịp điệu (pacing analysis)**.

---

## 5. Xử lý sự cố

**① Thiếu FFmpeg**
Triệu chứng: lỗi `ffmpeg not found` khi cắt/xuất video.
→ Cài FFmpeg (mục 1) rồi **mở lại** cửa sổ dòng lệnh và chạy lại. Kiểm tra bằng `ffmpeg -version`.

**② Tải model thủ công**
Nếu tải tự động thất bại (proxy/firewall), tải bằng trình duyệt rồi đặt đúng chỗ:
- Whisper small → **`backend/models/small.pt`** (~461 MB)
  `https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt`
- YuNet face model → **`backend/models/face_detection_yunet_2023mar.onnx`** (232589 bytes)
  `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`

Sau đó chạy lại bình thường. Nếu file đã tồn tại và đủ lớn, setup sẽ tự bỏ qua bước tải.

> ⚠️ Model Whisper **phải** nằm ở `backend/models/small.pt` — đó là vị trí
> `main.py` tìm khi gọi `whisper.load_model("small", download_root=backend/models)`.

**③ Đổi cổng (mặc định 8000)**
Cổng 8000 đang bị chiếm → sửa dòng cuối `backend/main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)   # đổi 8000 -> ví dụ 8080
```
(Nhớ đổi cả URL bạn mở trên trình duyệt và URL chia sẻ.)

**④ Proxy công ty**
- Bước cài pip và tải model đã **tự bắt** tình huống proxy (chứng chỉ tự ký) bằng
  cách tắt kiểm tra SSL — không cần cấu hình gì thêm.
- Nếu mạng công ty **chặn hẳn** các tên miền tải model, hãy tải model thủ công theo mục ② (dùng máy/mạng khác, ví dụ 4G điện thoại), rồi copy file vào.
- Không hardcode proxy/credentials ở đâu cả — tất cả đi qua cấu hình sẵn có của máy đích.

---

## 6. Cấu trúc gói

```
ClipForge-Portable/
├─ video-clipper.html              # Giao diện single-file (được phục vụ tại "/")
├─ setup_and_run.py                # Bộ cài + launcher chính
├─ run.bat                         # Chạy 1 cú nhấp trên Windows
├─ run.sh                          # Chạy trên Linux/macOS
├─ OPTIONAL_share_tunnel.bat       # (Tuỳ chọn) chia sẻ qua Cloudflare tunnel
├─ README.md                       # File này
└─ backend/
   ├─ main.py                      # FastAPI: phục vụ UI + API
   ├─ requirements.txt             # Danh sách thư viện
   └─ models/
      └─ face_detection_yunet_2023mar.onnx   # Model phát hiện mặt (232589 bytes)
```
> Model Whisper `small.pt` (~461 MB) **không** nằm trong gói để giữ zip nhỏ gọn —
> `setup_and_run.py` sẽ tự tải về `backend/models/small.pt` ở lần chạy đầu.
> Các thư mục `videos/`, `clips/`, `thumbnails/` sẽ tự được tạo khi chạy.

# Nhận diện chữ số viết tay bằng CNN

Bài lab dùng MNIST 28 × 28 và CNN gồm một lớp tích chập, Max Pooling, Flatten, Dense và Softmax. Notebook chia dữ liệu train của MNIST thành 54.000 ảnh huấn luyện và 6.000 ảnh validation. Checkpoint được chọn theo `val_loss`. Tập 10.000 ảnh test dành cho đánh giá cuối.

## Chạy trong PowerShell

Giải nén gói ở thư mục gốc của dự án để có thư mục `E:\DataScienceLab\DataScienceLab\mnist-cnn`. Các lệnh dưới đây dùng Python 3.12.

```powershell
Set-Location -LiteralPath 'E:\DataScienceLab\DataScienceLab\mnist-cnn'
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py --epochs 5
.\.venv\Scripts\python.exe app.py
```

Keras tải dữ liệu MNIST khi chạy huấn luyện lần đầu. Mở `http://127.0.0.1:5000`, vẽ một chữ số rồi bấm **Dự đoán**. Mô hình đã huấn luyện được lưu tại `artifacts/best_model.keras`.

Để chạy demo ngay với checkpoint có sẵn, tải [best_model.keras](https://chatgpt.com/library?file_id=libfile_8d7b1a4198e88191a7ba1f6894a756aa) và đặt cạnh `app.py`. Sau khi cài thư viện, chạy `app.py`. Ứng dụng ưu tiên checkpoint trong `artifacts/` nếu đã có kết quả huấn luyện mới.

Để làm bài theo từng bước trong PyCharm, mở `handwritten-digit-recognition.ipynb` và chọn interpreter `.venv`. Notebook huấn luyện mô hình, tính accuracy trên test và hiển thị một số dự đoán. Có thể đổi `EPOCHS` thành 1 để chạy thử nhanh.

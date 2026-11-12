# Data Science Labs

Bản thực hành dựa trên [nd-hung/data-science-labs](https://github.com/nd-hung/data-science-labs/tree/main/labs). Thư mục `data/` chứa 5 tệp dữ liệu gốc mà các notebook sử dụng.

| Bài | Nội dung |
| --- | --- |
| 01 · Pandas | Đọc bảng điểm Excel, kiểm tra dữ liệu thiếu, lọc theo ngành, tạo nhãn đậu/rớt và xuất CSV. |
| 01 · Matplotlib | Vẽ phổ điểm, so sánh nhóm ngành, vẽ đường giá vàng và vẽ biểu đồ phân tán cho giờ học, điểm thi và dữ liệu Iris. |
| 04 · Phân loại | Dự đoán đậu/rớt với Logistic Regression. Phân loại email spam với Multinomial Naive Bayes và đánh giá trên tập kiểm thử. |
| 05 · Mạng nơ-ron | Nhận diện chữ số viết tay với `MLPClassifier` và đo độ chính xác trên tập kiểm thử. |
| 06 · Phân cụm | Dùng K-means để phân cụm Iris và đánh giá với silhouette, Adjusted Rand Index và độ thuần cụm. |

## Chạy trên Windows

Mở thư mục dự án trong PyCharm hoặc chuyển đến thư mục bằng PowerShell. Dùng Python 3.11 trở lên để tạo môi trường và chạy toàn bộ notebook bằng các lệnh sau.

```powershell
Set-Location -LiteralPath 'E:\DataScienceLab\DataScienceLab'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_labs.py
```

`run_labs.py` chạy lần lượt cả 5 notebook và lưu kết quả ngay trong các tệp `.ipynb`. Bạn cũng có thể mở từng notebook trong PyCharm và chọn **Run All** với môi trường `.venv`.

Bài Pandas ghi CSV vào `outputs/final_grades_updated.csv`. Bài Matplotlib lưu biểu đồ vào `labs/01-working-with-data/figs/`.

## Thay đổi so với repo gốc

- Hoàn thành phần biểu đồ đường cho dữ liệu giá vàng.
- Tính độ chính xác của mô hình spam từ nhãn dự đoán và bổ sung ma trận nhầm lẫn.
- Đặt seed cho phép chia dữ liệu của Logistic Regression. Bài MLP chia dữ liệu theo tỷ lệ của từng lớp.
- Dùng đúng tên K-means cho notebook trong thư mục Clustering. Báo cáo độ thuần cụm khi gán mỗi cụm cho loài chiếm đa số.

# BaoCao_ECommerce
 Báo cáo Thực hành cuối kỳ môn Công nghệ Thương mại Điện tử
 
 Đề tài: Dự báo xu hướng giá chứng khoán dựa trên phương pháp mô hình hóa thống kê
 
GVHD: Ths. Đỗ Duy Thanh

HVTH nhóm 01:
1. Nguyễn Thị Mỹ Hải - 210104006
2. Đỗ Chí Bảo - 210104002
3. Hoàng Minh Khiêm - 210104013

Thư mục này bao gồm 3 thư mục:
1. Code: 
- Code xử lý dữ liệu tạo dataset tin tức (đã được gán nhãn tăng giảm)
- Code chạy huấn luyện các mô hình dự báo (RNN, GRU, LSTM, BiLSTM, BGRU)
- Code chạy mô hình đã huấn luyện để dự báo xu hướng tăng giảm của giá chứng khoán dựa vào tin tức đầu vào trong ngày.
2. Data
- Data tin tức được craw từ trang cafef.vn: TinTucGocVNIndex
- Data giá chứng khoán từ trang vn.investing.com: VnIndex
- Data sau xử lý (NewsAll_Flag): bộ data tin tức đã được xử lý tách từ, loại bỏ từ dừng, từ ít nghĩa và gán nhãn tăng giảm theo giá chứng khoán.

*** Lưu ý: Riêng data TinTucGocVNIndex và NewsAll_Flag nhóm chuyển sang thư mục Dataset thuộc dashboard để sử dụng cho report dashboard.
Vậy nên khi chạy code nếu dùng đến 2 data này vui lòng lưu ý đường dẫn thư mục data.

3. Dashboard: Bao gồm 2 phần chính cần lưu ý
- Code chạy dashboard: app.py
- Dataset để chạy các report trên dashboard.

Để khởi chạy dashboard với thư viện Dash của Python vui lòng thực hiện các bước sau:
B1: Setup môi trường
Sử dụng command Prompt chạy các dòng lệnh sau:

pip install -r requirements.txt

mkdir Dashboard 

cd Dashboard  

python -m venv venv 

venv\Scripts\activate

B2: Copy thư mục assets, dataset và file app, requirements vào thư mục Dashboard mới tạo

B2: Khởi chạy app.py để mở dashboard bằng dòng lệnh sau: Python app.py

B3: Copy link http://localhost:8050/ và mở trên trình duyệt web để mở dashboard 

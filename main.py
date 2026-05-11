import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cấu hình trang web
st.set_page_config(page_title="Dự đoán Độ ưu tiên Đơn hàng", layout="centered")

st.title("🚚 Hệ thống Dự đoán Priority Đơn hàng")
st.write("Ứng dụng sử dụng thuật toán **Perceptron** để phân loại độ ưu tiên.")

# 1. Tải dữ liệu
@st.cache_resource
def train_model():
    try:
        # Đọc dữ liệu (đảm bảo file dulieu.csv nằm cùng thư mục)
        data = pd.read_csv("dulieu.csv")
        data.columns = ['OrderSize', 'Distance', 'Fragile', 'Priority']
        
        X = data[['OrderSize', 'Distance', 'Fragile']]
        y = data['Priority']

        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Chuẩn hóa
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        # Huấn luyện
        model = Perceptron(max_iter=1000, eta0=0.05, random_state=0)
        model.fit(X_train_std, y_train)
        
        # Tính độ chính xác
        y_pred = model.predict(X_test_std)
        acc = accuracy_score(y_test, y_pred)
        
        return model, sc, acc
    except FileNotFoundError:
        return None, None, None

model, sc, acc = train_model()

if model is None:
    st.error("❌ Không tìm thấy file 'dulieu.csv'. Vui lòng upload file lên GitHub cùng với code.")
else:
    # Hiển thị độ chính xác trong Sidebar
    st.sidebar.header("Thông tin mô hình")
    st.sidebar.write(f"Độ chính xác: **{acc*100:.2f}%**")

    # 2. Giao diện dự đoán
    st.subheader("Nhập thông tin đơn hàng mới")
    
    col1, col2 = st.columns(2)
    
    with col1:
        order_size = st.number_input("Kích thước đơn hàng (OrderSize):", min_value=1, value=20)
        distance = st.number_input("Khoảng cách (Distance):", min_value=1, value=50)
        
    with col2:
        fragile = st.selectbox("Hàng dễ vỡ? (Fragile):", options=[0, 1], format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")

    if st.button("Dự đoán ngay"):
        # Chuẩn bị dữ liệu input
        input_data = np.array([[order_size, distance, fragile]])
        input_std = sc.transform(input_data)
        
        # Dự đoán
        prediction = model.predict(input_std)
        
        # Hiển thị kết quả
        st.divider()
        result_text = "🔴 ƯU TIÊN CAO" if prediction[0] == 1 else "🔵 Bình thường"
        st.success(f"Kết quả dự đoán Priority: **{prediction[0]}** ({result_text})")

    # Hiển thị dữ liệu mẫu nếu muốn
    if st.checkbox("Xem dữ liệu huấn luyện mẫu"):
        df = pd.read_csv("dulieu.csv")
        st.dataframe(df.head())
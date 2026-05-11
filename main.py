import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cấu hình trang web
st.set_page_config(page_title="Dự đoán Độ ưu tiên Đơn hàng", page_icon="📦")

st.title("🚚 Hệ thống Dự đoán Priority Đơn hàng")
st.write("Mô hình sử dụng thuật toán **Perceptron** để phân loại đơn hàng vào 3 mức độ ưu tiên.")

# 1. Hàm huấn luyện mô hình (Dùng cache để không phải train lại mỗi khi bấm nút)
@st.cache_resource
def train_model():
    try:
        # Đọc dữ liệu
        data = pd.read_csv("dulieu.csv")
        data.columns = ['OrderSize', 'Distance', 'Fragile', 'Priority']
        
        X = data[['OrderSize', 'Distance', 'Fragile']]
        y = data['Priority']

        # Chia dữ liệu (70% train, 30% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Chuẩn hóa dữ liệu (Rất quan trọng với Perceptron)
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        # Khởi tạo và huấn luyện Perceptron (Tự động xử lý đa lớp)
        model = Perceptron(max_iter=1000, eta0=0.05, random_state=0)
        model.fit(X_train_std, y_train)
        
        # Tính độ chính xác
        y_pred = model.predict(X_test_std)
        acc = accuracy_score(y_test, y_pred)
        
        return model, sc, acc
    except Exception as e:
        return None, None, str(e)

# Chạy hàm train
model, sc, acc_info = train_model()

if model is None:
    st.error(f"❌ Lỗi: {acc_info}")
    st.info("💡 Đảm bảo file 'dulieu.csv' nằm cùng thư mục với file code này.")
else:
    # Hiển thị thông số ở Sidebar
    st.sidebar.header("📊 Thông số mô hình")
    st.sidebar.metric("Độ chính xác (Accuracy)", f"{acc_info*100:.1f}%")
    st.sidebar.markdown("""
    **Quy ước Priority:**
    - 0: Thấp (Low)
    - 1: Trung bình (Medium)
    - 2: Cao (High)
    """)

    # 2. Giao diện nhập liệu
    st.subheader("📝 Nhập thông số đơn hàng")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            order_size = st.number_input("Số lượng (OrderSize):", min_value=0, value=20)
        with col2:
            distance = st.number_input("Khoảng cách (Distance):", min_value=0, value=50)
        with col3:
            fragile = st.selectbox("Dễ vỡ? (Fragile):", options=[0, 1], 
                                  format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")

    # 3. Dự đoán và Hiển thị kết quả
    if st.button("🚀 Dự đoán mức độ ưu tiên"):
        # Chuẩn hóa input theo scaler đã train
        input_data = np.array([[order_size, distance, fragile]])
        input_std = sc.transform(input_data)
        
        # Dự đoán
        prediction = model.predict(input_std)[0]
        
        # Mapping kết quả để hiển thị cho đẹp
        result_map = {
            0: ("🔵 THẤP (Low)", "Màu xanh dương biểu thị đơn hàng bình thường."),
            1: ("🟡 TRUNG BÌNH (Medium)", "Màu vàng biểu thị đơn hàng cần lưu ý."),
            2: ("🔴 CAO (High)", "Màu đỏ biểu thị đơn hàng cần xử lý gấp!")
        }
        
        label, note = result_map.get(prediction, ("Không xác định", ""))
        
        st.divider()
        st.header(f"Kết quả: {label}")
        st.info(f"Giải thích: {note}")

    # Xem dữ liệu gốc
    if st.checkbox("Hiển thị dữ liệu mẫu từ dulieu.csv"):
        df_view = pd.read_csv("dulieu.csv")
        st.dataframe(df_view.head(10))

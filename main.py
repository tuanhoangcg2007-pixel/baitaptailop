import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier # Thuật toán mạnh hơn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Cấu hình trang
st.set_page_config(page_title="AI Dự đoán Đơn hàng", page_icon="🎯")

st.title("🚀 Hệ thống Dự đoán Priority (Nâng cao)")
st.write("Mô hình đã được nâng cấp lên **Random Forest** để đạt độ chính xác cao nhất.")

# 1. Hàm huấn luyện mô hình
@st.cache_resource
def train_high_accuracy_model():
    try:
        # Đọc dữ liệu
        data = pd.read_csv("dulieu.csv")
        # Giả định file có 4 cột: OrderSize, Distance, Fragile, Priority
        data.columns = ['OrderSize', 'Distance', 'Fragile', 'Priority']
        
        X = data[['OrderSize', 'Distance', 'Fragile']]
        y = data['Priority']

        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Sử dụng RandomForestClassifier thay cho Perceptron
        # n_estimators=100 là tạo ra 100 "cây" quyết định để lấy ý kiến số đông
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Tính độ chính xác
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        return model, acc
    except Exception as e:
        return None, str(e)

model, acc_info = train_high_accuracy_model()

if model is None:
    st.error(f"❌ Lỗi tải dữ liệu: {acc_info}")
else:
    # Hiển thị độ chính xác trên web
    st.sidebar.success(f"✅ Độ chính xác mô hình: {acc_info*100:.2f}%")
    st.sidebar.markdown("""
    **Mức độ ưu tiên:**
    - 0: 🔵 Thấp (Low)
    - 1: 🟡 Trung bình (Medium)
    - 2: 🔴 Cao (High)
    """)

    # 2. Giao diện nhập liệu
    st.subheader("📊 Nhập thông số để dự đoán")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        size = st.number_input("Kích thước đơn:", min_value=0.0, step=0.1, value=25.0)
    with col2:
        dist = st.number_input("Khoảng cách (km):", min_value=0.0, step=0.1, value=10.0)
    with col3:
        frag = st.selectbox("Dễ vỡ?", [0, 1], format_func=lambda x: "Có" if x == 1 else "Không")

    if st.button("Tính toán Priority"):
        # Dự đoán
        input_data = np.array([[size, dist, frag]])
        prediction = model.predict(input_data)[0]
        
        # Hiển thị kết quả chuyên nghiệp
        st.divider()
        if prediction == 2:
            st.error(f"### KẾT QUẢ: MỨC ĐỘ CAO (HIGH - 2)")
            st.toast("Cần xử lý ngay lập tức!", icon="🔥")
        elif prediction == 1:
            st.warning(f"### KẾT QUẢ: MỨC ĐỘ TRUNG BÌNH (MEDIUM - 1)")
        else:
            st.info(f"### KẾT QUẢ: MỨC ĐỘ THẤP (LOW - 0)")

    # 3. Phân tích tầm quan trọng của các yếu tố
    if st.checkbox("Xem yếu tố nào ảnh hưởng nhất đến Priority?"):
        importances = pd.Series(model.feature_importances_, index=['Kích thước', 'Khoảng cách', 'Dễ vỡ'])
        st.bar_chart(importances)

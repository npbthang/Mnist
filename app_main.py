import streamlit as st

# Cấu hình trang
st.set_page_config(
    page_title="Ứng dụng Đa Năng",
    page_icon="🚀",
    layout="wide",
)

# Tạo layout hai cột: Sidebar bên trái, nội dung bên phải
col1, col2 = st.columns([1, 3])

with col1:
    # Sidebar với thiết kế mới
    st.markdown(
        """
        <style>
            .sidebar-content {
                background: linear-gradient(135deg, #232526, #414345);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }
            .sidebar-content h2 {
                color: #FFFFFF;
            }
            .sidebar-content p {
                color: #D3D3D3;
            }
        </style>
        <div class="sidebar-content">
            <h2>🌟 Menu</h2>
            <p>Chọn một ứng dụng để trải nghiệm!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    # Tiêu đề chính
    st.title("🚀 Ứng dụng Đa Năng với Streamlit")

    # Danh sách ứng dụng dạng Grid
    st.markdown(
        """
        <style>
            .app-container {
                background: linear-gradient(135deg, #2C3E50, #4CA1AF);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
            }
            .app-title {
                color: #FFC107;
                font-size: 22px;
                font-weight: bold;
            }
            .app-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                list-style: none;
                padding: 0;
            }
            .app-item {
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 8px;
                font-size: 16px;
                color: #FFFFFF;
                font-weight: 500;
            }
            .app-item strong {
                color: #FFEB3B;
            }
        </style>
        <div class="app-container">
            <h3 class="app-title">📋 Danh sách Ứng Dụng</h3>
            <ul class="app-grid">
                <li class="app-item">📊 <strong>Linear Regression</strong><br> Phân tích hồi quy tuyến tính.</li>
                <li class="app-item">🔢 <strong>MNIST Classification</strong><br> Nhận dạng chữ số viết tay.</li>
                <li class="app-item">📌 <strong>Clustering Algorithms</strong><br> Thuật toán phân cụm dữ liệu.</li>
                <li class="app-item">📉 <strong>PCA & t-SNE MNIST</strong><br> Giảm chiều dữ liệu với PCA và t-SNE.</li>
                <li class="app-item">🧠 <strong>Neural Network MNIST</strong><br> Mô hình mạng nơ-ron nhân tạo.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    "<p style='text-align: center; color: #B0BEC5; font-size: 12px;'>🚀 Được phát triển với Streamlit</p>",
    unsafe_allow_html=True
)

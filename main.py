import streamlit as st
from PIL import Image
from src.predict import predict_flower  # Import hàm dự đoán

# Tiêu đề cửa sổ
st.set_page_config(page_title="FLOWER_CLASSIFICATION", layout="wide")

# CSS giao diện nà
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #FDF2F8, #FAF5FF, #EFF6FF);
    color: #E60076;
    font-family: "Arial", sans-serif;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stToolbar"] {display: none;}
.block-container {
    max-width: 95% !important;
    padding: 1rem 0 2rem 0;
}
h3 {
    color: #00d4ff;
    font-weight: 400;
    margin-bottom: 2rem;
    padding-top: 5rem;
}
.subtitle {
    text-align: center;
    color: #6A5ACD;
    font-size: 16px;
    margin-bottom: 2rem;
}
/* Card trắng bo góc ôm nội dung */
.card-box {
    background: white;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    padding: 10px 20px;
    margin: 10px 0;
}

/* Tiêu đề từng khối */
.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #e6007e;
    display: flex;
    align-items: center;
    gap: 6px;
}
.section-subtitle {
    color: #e6007e;
    font-size: 14px;
    margin-bottom: 15px;
}

/* Màu info khi chưa upload */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Ba khối dưới cùng */
.feature-card {
    background: #fff3fa;
    border-radius: 18px;
    text-align: center;
    padding: 25px 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: all 0.2s ease;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 5px 12px rgba(0,0,0,0.1);
}
.feature-icon {
    font-size: 26px;
    background: #ff2fa2;
    color: white;
    border-radius: 50%;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px auto;
}
.feature-title {
    font-weight: 600;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
# Tạo 3 cột: trái - giữa (chứa nội dung) - phải
left, center, right = st.columns([4, 3, 4])  # tổng là 8 phần → khối giữa chiếm 25% chiều ngang

with center:
    col1, col2 = st.columns([1, 9])  # chia nhỏ trong khối giữa

    with col1:
        st.markdown("<div>", unsafe_allow_html=True)
        st.image("icon/icon_flower.png", width=30)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
        st.markdown("<h3 >Nhận Diện Loài Hoa</h3>", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Upload ảnh hoa để nhận diện và phân loại loài hoa bằng công nghệ AI</p>", unsafe_allow_html=True)

# ====== KHỐI CHÍNH: Upload + Kết quả ======
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="card-box" style="
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        text-align: center;
    ">
        <div class="section-title" style="font-size: 20px; font-weight: bold; color: #ff007f;">
            🌸 Upload Ảnh
        </div>
        <div class="section-subtitle">
            Kéo thả hoặc chọn ảnh hoa để nhận diện
        </div>
    """, unsafe_allow_html=True)

    # ✅ uploader đặt giữa khung div
    uploaded_file = st.file_uploader(" ", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    # hiển thị ảnh nếu có
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh bạn đã chọn", use_container_width=True)
    else:
        st.markdown('<p style="color:#888;">📁 Kéo thả ảnh vào đây hoặc chọn file</p>', unsafe_allow_html=True)

    # đóng khung upload-box + card-box
    st.markdown("""
        </div> <!-- đóng upload-box -->
    </div> <!-- đóng card-box -->
    """, unsafe_allow_html=True)
# --- Cột phải: Kết quả ---
with col2:
    st.markdown(f"""
    <div class="card-box">
        <div class="section-title">🌼 Kết Quả Phân Loại</div>
        <div class="section-subtitle">Kết quả nhận diện loài hoa từ ảnh của bạn</div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        # Dự đoán trực tiếp từ ảnh đã mở, KHÔNG mở lại
        predicted_class, confidence = predict_flower(image)

        st.success(f"🌷 Đây là hoa: **{predicted_class.upper()}** ({confidence:.2%} tự tin)")
    else:
        st.info("Upload ảnh để xem kết quả phân loại.")

    # đóng khung
    st.markdown("</div>", unsafe_allow_html=True)

# ====== KHỐI DƯỚI: 3 TÍNH NĂNG ======
st.markdown("<br><br>", unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📤</div>
        <div class="feature-title">Upload Dễ Dàng</div>
        <p>Kéo thả hoặc chọn ảnh từ thiết bị của bạn</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">AI Thông Minh</div>
        <p>Công nghệ AI nhận diện chính xác các loài hoa</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🖼️</div>
        <div class="feature-title">Kết Quả Chi Tiết</div>
        <p>Hiển thị tên, độ chính xác và thông tin loài hoa</p>
    </div>
    """, unsafe_allow_html=True)
# KHỐI DƯỚI: KẾT QUẢ BỘ TEST
st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.markdown("<h4 style='color:#e6007e;'>📊 Kết Quả Bộ Test Mô Hình</h4>", unsafe_allow_html=True)

col6, col7 = st.columns(2)

with col6:
    st.image("outputs/accuracy_plot.png", caption="Biểu đồ Accuracy", use_container_width=True)
    st.image("outputs/loss_plot.png", caption="Biểu đồ Loss", use_container_width=True)

with col7:
    st.image("outputs/confusion_matrix.png", caption="Ma trận nhầm lẫn", use_container_width=True)
    try:
        with open("outputs/classification_report.txt", "r", encoding="utf-8") as f:
            report = f.read()
        st.markdown("Báo cáo phân loại:")
        st.code(report, language="text")
    except:
        st.warning("Không tìm thấy file classification_report.txt")
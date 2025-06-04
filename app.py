import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import av
import time
from PIL import Image
import os
import gdown
import threading
import queue
from pathlib import Path
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "best_modelnew.h5.keras"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1nB_Sr_jnm0HmMSC4ISf2bFOYPGLW15Bc"
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 255, 255),  # Yellow
    'fear': (128, 0, 128),     # Purple
    'happy': (0, 128, 0),      # Green
    'sad': (255, 0, 0),        # Blue
    'surprise': (255, 165, 0), # Orange
    'neutral': (128, 128, 128) # Gray
}

class ModelManager:
    """Quản lý tải và cache mô hình"""
    
    @staticmethod
    @st.cache_resource
    def load_emotion_model_from_file(model_path: str):
        """Tải mô hình từ file path cụ thể"""
        try:
            model = load_model(model_path, compile=False)
            logger.info(f"Model loaded successfully from: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise e
    
    @staticmethod
    def check_and_load_model():
        """Kiểm tra và tải mô hình với các phương thức khác nhau"""
        model_path = Path(MODEL_PATH)
        
        # Kiểm tra file tồn tại và tải trực tiếp
        if model_path.exists():
            try:
                model = ModelManager.load_emotion_model_from_file(str(model_path))
                st.success(f"✅ Mô hình đã được tải thành công từ: {model_path}")
                return model
            except Exception as e:
                st.error(f"❌ Lỗi khi tải mô hình từ file có sẵn: {e}")
                # Xóa file lỗi
                try:
                    model_path.unlink()
                except:
                    pass
        
        # Tải từ Google Drive
        return ModelManager._download_from_drive()
    
    @staticmethod
    def _download_from_drive():
        """Tải mô hình từ Google Drive"""
        try:
            with st.spinner("🔄 Đang tải mô hình từ Google Drive..."):
                gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
                
            if Path(MODEL_PATH).exists():
                model = ModelManager.load_emotion_model_from_file(MODEL_PATH)
                st.success("✅ Tải mô hình từ Google Drive thành công!")
                return model
            else:
                raise FileNotFoundError("Không thể tải file từ Google Drive.")
                
        except Exception as e:
            st.error(f"❌ Lỗi khi tải từ Google Drive: {e}")
            st.info("💡 Bạn có thể thử các cách sau:")
            st.markdown("""
            1. **Kiểm tra kết nối internet**
            2. **Đảm bảo Google Drive link hợp lệ**
            3. **Tải file mô hình thủ công ở phần dưới**
            """)
            return None
    
    @staticmethod
    def handle_manual_upload(uploaded_model):
        """Xử lý tải mô hình thủ công"""
        if uploaded_model is not None:
            try:
                # Lưu file tạm thời
                temp_path = f"temp_{MODEL_PATH}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                
                # Thử tải mô hình
                model = ModelManager.load_emotion_model_from_file(temp_path)
                
                # Nếu thành công, chuyển sang tên chính thức
                if Path(MODEL_PATH).exists():
                    Path(MODEL_PATH).unlink()
                Path(temp_path).rename(MODEL_PATH)
                
                st.success("✅ File mô hình đã được tải lên và khởi tạo thành công!")
                st.balloons()
                return model
                
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý file mô hình: {e}")
                # Xóa file tạm nếu lỗi
                if Path(f"temp_{MODEL_PATH}").exists():
                    Path(f"temp_{MODEL_PATH}").unlink()
                return None
        
        return None

class ImageProcessor:
    """Xử lý hình ảnh và dự đoán cảm xúc"""
    
    @staticmethod
    def preprocess_face(face_img):
        """Tiền xử lý ảnh khuôn mặt với error handling"""
        try:
            if face_img is None or face_img.size == 0:
                return None
                
            # Chuyển đổi màu sắc
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Cải thiện độ tương phản
            lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Resize và normalize
            face_img = cv2.resize(face_img, (299, 299))
            face_img = face_img.astype('float32')
            face_img = (face_img - 127.5) / 127.5
            face_img = np.expand_dims(face_img, axis=0)
            
            return face_img
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None
    
    @staticmethod
    def predict_emotion(face_img, model):
        """Dự đoán cảm xúc với confidence threshold"""
        try:
            if model is None:
                return None, 0.0
                
            processed_img = ImageProcessor.preprocess_face(face_img)
            if processed_img is None:
                return None, 0.0
            
            predictions = model.predict(processed_img, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Chỉ trả về kết quả nếu confidence > threshold
            if confidence > 0.3:  # Threshold để lọc prediction yếu
                return EMOTION_LABELS[predicted_class], confidence
            else:
                return None, confidence
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.0

class OptimizedVideoProcessor(VideoProcessorBase):
    """Video processor được tối ưu hóa với xử lý no-face detection"""
    
    def __init__(self, model):
        self.model = model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Prediction caching và tracking
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_interval = 0.3
        self.no_face_start_time = None  # Thời điểm bắt đầu không có face
        self.no_face_threshold = 1.0    # Sau 1 giây không có face thì clear prediction
        
        # Frame skipping for better performance
        self.frame_count = 0
        self.process_every_n_frames = 3
        
        logger.info("VideoProcessor initialized successfully")
    
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            current_time = time.time()
            
            # Tính FPS
            self.fps_counter += 1
            if current_time - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = current_time
            
            output_img = img.copy()
            
            # LUÔN LUÔN kiểm tra face detection để responsive hơn
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces với parameters tối ưu
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=3, 
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            faces_detected = len(faces) > 0
            
            if faces_detected:
                # CÓ FACE - Reset no-face timer
                self.no_face_start_time = None
                
                # Chỉ predict emotion theo interval để tối ưu performance
                should_process = (self.frame_count % self.process_every_n_frames == 0)
                should_predict = (current_time - self.last_prediction_time) >= self.prediction_interval
                
                if should_process and should_predict:
                    # Chọn khuôn mặt lớn nhất (gần camera nhất)
                    largest_face = max(faces, key=lambda face: face[2] * face[3])
                    x, y, w, h = largest_face
                    
                    # Mở rộng vùng face một chút
                    margin = int(max(w, h) * 0.1)
                    face_x = max(0, x - margin)
                    face_y = max(0, y - margin)
                    face_w = min(img.shape[1] - face_x, w + 2 * margin)
                    face_h = min(img.shape[0] - face_y, h + 2 * margin)
                    
                    face_img = img[face_y:face_y+face_h, face_x:face_x+face_w]
                    
                    if face_img.size > 0:
                        emotion, confidence = ImageProcessor.predict_emotion(face_img, self.model)
                        if emotion is not None:
                            self.last_prediction = (emotion, confidence, largest_face)
                            self.last_prediction_time = current_time
            else:
                # KHÔNG CÓ FACE
                if self.no_face_start_time is None:
                    # Bắt đầu đếm thời gian không có face
                    self.no_face_start_time = current_time
                elif (current_time - self.no_face_start_time) > self.no_face_threshold:
                    # Đã quá lâu không có face, XÓA prediction cũ
                    self.last_prediction = None
            
            # Vẽ kết quả
            if faces_detected or (self.last_prediction is not None and 
                                 self.no_face_start_time is None):
                # Có face hiện tại hoặc vừa mới mất face (chưa quá threshold)
                if self.last_prediction is not None:
                    emotion, confidence, (x, y, w, h) = self.last_prediction
                    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                    
                    # Vẽ rectangle
                    cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 3)
                    
                    # Vẽ text với background
                    text = f"{emotion}: {confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    
                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # Background cho text
                    bg_y = max(0, y - 10)
                    cv2.rectangle(output_img, 
                                (x, bg_y - text_h - baseline), 
                                (x + text_w, bg_y), 
                                color, -1)
                    
                    # Text
                    cv2.putText(output_img, text, (x, bg_y - baseline), 
                              font, font_scale, (255, 255, 255), thickness)
                    
                    label = f'{emotion}: {confidence:.2f} | FPS: {self.current_fps:.1f}'
                else:
                    label = f'FPS: {self.current_fps:.1f}'
            else:
                # Không có face được phát hiện
                # Hiển thị thông báo "Không phát hiện khuôn mặt"
                text = "no face detected"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                color = (0, 0, 255)  # Màu đỏ
                
                # Tính toán vị trí để đặt text ở giữa màn hình
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                img_h, img_w = output_img.shape[:2]
                text_x = (img_w - text_w) // 2
                text_y = (img_h + text_h) // 2
                
                # Background cho text
                cv2.rectangle(output_img, 
                            (text_x - 10, text_y - text_h - baseline - 10), 
                            (text_x + text_w + 10, text_y + baseline + 10), 
                            (0, 0, 0), -1)
                
                # Text
                cv2.putText(output_img, text, (text_x, text_y), 
                          font, font_scale, color, thickness)
                
                label = f'Khong phat hien khuon mat | FPS: {self.current_fps:.1f}'
            
            # Update session state
            if 'label' in st.session_state:
                st.session_state['label'] = label
            
            return av.VideoFrame.from_ndarray(output_img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return frame

def get_webrtc_configuration():
    """Cấu hình WebRTC tối ưu để khắc phục lỗi connection"""
    
    # Cấu hình STUN servers mạnh mẽ hơn
    ice_servers = [
        # Google STUN servers - Ổn định nhất
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"},
        
        # OpenRelay TURN servers - Free và ổn định
        {
            "urls": "turn:relay1.expressturn.com:3480",
            "username": "000000002063846457",
            "credential": "IjDTiMpkfaNqnGhlGFSRC7GMJpU="
        },
      
        
        # Backup STUN servers
        {"urls": "stun:stun.stunprotocol.org:3478"},
        {"urls": "stun:stun.mozilla.org"},
    ]
    
    return RTCConfiguration({
        "iceServers": ice_servers,
        "iceTransportPolicy": "all",
        "bundlePolicy": "max-bundle",
        "rtcpMuxPolicy": "require", 
        "iceCandidatePoolSize": 10,
        "iceConnectionReceiveTimeout": 30000,  # 30 seconds
        "iceGatheringTimeout": 10000,         # 10 seconds
    })
def process_uploaded_image(uploaded_file, model):
    """Xử lý hình ảnh được tải lên"""
    try:
        image = Image.open(uploaded_file)
        
        # Hiển thị hình ảnh gốc
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Hình ảnh gốc")
            st.image(image, use_column_width=True)
        
        # Xử lý
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        with col2:
            st.subheader("🎭 Kết quả nhận diện")
            
            if len(faces) == 0:
                st.warning("⚠️ Không phát hiện khuôn mặt trong hình ảnh")
                return
            
            # Xử lý tất cả các khuôn mặt
            results = []
            result_img = img_array.copy()
            
            for i, (x, y, w, h) in enumerate(faces):
                face_img = img_array[y:y+h, x:x+w]
                emotion, confidence = ImageProcessor.predict_emotion(face_img, model)
                
                if emotion is not None:
                    results.append((emotion, confidence))
                    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                    
                    # Vẽ rectangle và text
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 3)
                    text = f"{emotion}: {confidence:.2f}"
                    cv2.putText(result_img, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Hiển thị kết quả
            if results:
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img_rgb, use_column_width=True)
                
                st.subheader("📊 Chi tiết kết quả:")
                for i, (emotion, confidence) in enumerate(results):
                    st.write(f"**Khuôn mặt {i+1}:** {emotion} ({confidence:.2%})")
            else:
                st.error("❌ Không thể dự đoán cảm xúc từ các khuôn mặt được phát hiện")
                
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý hình ảnh: {e}")
        logger.error(f"Image processing error: {e}")

def main():
    # Cấu hình trang
    st.set_page_config(
        page_title="Emotion Recognition",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    def hide_streamlit_style():
    """Ẩn menu và các thành phần không cần thiết của Streamlit"""
    hide_st_style = """
            <style>
            /* Ẩn menu hamburger */
            #MainMenu {visibility: hidden;}
            
            /* Ẩn header mặc định */
            header {visibility: hidden;}
            
            /* Ẩn footer "Made with Streamlit" */
            footer {visibility: hidden;}
            
            /* Ẩn nút Deploy (nếu có) */
            .stDeployButton {display: none;}
            
            /* Ẩn toàn bộ toolbar phía trên */
            .stAppToolbar {display: none;}
            
            /* Tùy chọn: Ẩn phần padding phía trên */
            .stAppHeader {display: none;}
            
            /* Tùy chọn: Điều chỉnh padding */
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
            }
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    # Header
    st.title("🎭 Nhận Diện Cảm Xúc Khuôn Mặt")
    st.markdown("---")
    
    # Sidebar thông tin
    with st.sidebar:
        st.header("ℹ️ Thông tin ứng dụng")
        st.info("""
        **Mô hình:**
        **Cảm xúc nhận diện:** 
        - 😠 Angry (Tức giận)
        - 🤢 Disgust (Ghê tởm)  
        - 😰 Fear (Sợ hãi)
        - 😊 Happy (Vui vẻ)
        - 😢 Sad (Buồn bã)
        - 😲 Surprise (Ngạc nhiên)
        - 😐 Neutral (Bình thường)
        """)
        
        st.header("⚙️ Cài đặt")
        if st.button("🔄 Tải lại mô hình", help="Xóa cache và tải lại mô hình"):
            st.cache_resource.clear()
            st.rerun()
        
        # Cài đặt WebRTC
        st.header("📡 Cài đặt Camera")
        video_quality = st.selectbox(
            "Chất lượng video",
            ["Thấp (320x240)", "Trung bình (640x480)", "Cao (1280x720)"],
            index=1,
            help="Chất lượng thấp hơn giúp kết nối ổn định hơn trên mạng chậm"
        )
        
        frame_rate = st.slider(
            "Tốc độ khung hình (FPS)",
            min_value=5,
            max_value=30,
            value=15,
            help="FPS thấp hơn giúp kết nối ổn định hơn"
        )
        
        # Network troubleshooting
        st.header("🔧 Khắc phục sự cố")
        if st.button("🔍 Kiểm tra kết nối"):
            st.info("""
            **Nếu gặp lỗi kết nối:**
            1. Thử chuyển sang chất lượng video thấp hơn
            2. Giảm FPS xuống 10-15
            3. Kiểm tra tường lửa của công ty/trường học
            4. Thử dùng VPN nếu mạng bị hạn chế
            5. Restart browser và clear cache
            """)
        
        # Hiển thị trạng thái model
        if Path(MODEL_PATH).exists():
            st.success("✅ Mô hình có sẵn")
        else:
            st.warning("⚠️ Chưa có mô hình")
    
    # Khởi tạo session state
    if 'label' not in st.session_state:
        st.session_state['label'] = '🔄 Đang khởi tạo...'
    if 'model_loaded' not in st.session_state:
        st.session_state['model_loaded'] = False
    
    # Tải mô hình
    model = ModelManager.check_and_load_model()
    
    # Nếu không có model, hiển thị phần upload
    if model is None:
        st.subheader("📤 Tải mô hình thủ công")
        st.info("Vui lòng tải file mô hình (.h5 hoặc .keras) để tiếp tục.")
        
        uploaded_model = st.file_uploader(
            "Chọn file mô hình", 
            type=["h5", "keras"],
            help="File mô hình phải có định dạng .h5 hoặc .keras",
            key="model_uploader"
        )
        
        if uploaded_model is not None:
            with st.spinner("🔄 Đang xử lý file mô hình..."):
                model = ModelManager.handle_manual_upload(uploaded_model)
                if model is not None:
                    st.session_state['model_loaded'] = True
                    st.rerun()
    
    if model is None:
        st.error("❌ Không thể tiếp tục do chưa có mô hình. Vui lòng tải mô hình ở trên.")
        st.stop()
    
    # Tabs cho các chức năng
    tab1, tab2 = st.tabs(["📷 Tải hình ảnh", "🎥 Camera trực tiếp"])
    
    with tab1:
        st.subheader("📤 Tải hình ảnh để dự đoán cảm xúc")
        st.write("Hỗ trợ định dạng: JPG, JPEG, PNG")
        
        uploaded_file = st.file_uploader(
            "Chọn hình ảnh", 
            type=["jpg", "jpeg", "png"],
            help="Hình ảnh nên chứa khuôn mặt rõ ràng để có kết quả tốt nhất",
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            process_uploaded_image(uploaded_file, model)
    
    with tab2:
        st.subheader("🎥 Nhận diện cảm xúc từ camera")
        st.write("Cho phép truy cập camera và bắt đầu nhận diện")
        
        # Cấu hình video dựa trên lựa chọn người dùng
        if video_quality == "Thấp (320x240)":
            video_constraints = {
                "width": {"ideal": 320},
                "height": {"ideal": 240},
            }
        elif video_quality == "Trung bình (640x480)":
            video_constraints = {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
            }
        else:  # Cao
            video_constraints = {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
            }
        
        video_constraints["frameRate"] = {"ideal": frame_rate, "max": 30}
        
        # Cảnh báo về mạng
        st.warning("""
        ⚠️ **Lưu ý về kết nối:**
        - Nếu gặp lỗi "connection taking longer", hãy thử giảm chất lượng video
        - Một số mạng công ty/trường học có thể chặn WebRTC
        - Mạng 4G đôi khi không ổn định với WebRTC
        """)
        
        # WebRTC Streamer với cấu hình cải thiện
        webrtc_ctx = webrtc_streamer(
            key="emotion-recognition-improved",
            video_processor_factory=lambda: OptimizedVideoProcessor(model),
            rtc_configuration=get_webrtc_configuration(),
            media_stream_constraints={
                "video": video_constraints,
                "audio": False
            },
            async_processing=True,
        )
        
        # Hiển thị trạng thái
        status_placeholder = st.empty()
        
        if webrtc_ctx.video_processor:
            with status_placeholder.container():
                st.success("✅ Camera đang hoạt động")
                if 'label' in st.session_state:
                    st.info(f"📊 **Trạng thái:** {st.session_state['label']}")
        else:
            with status_placeholder.container():
                st.warning("⚠️ Camera chưa được kích hoạt. Nhấn 'Start' để bắt đầu.")
                
                # Thêm hướng dẫn troubleshooting
                with st.expander("🔧 Hướng dẫn khắc phục lỗi kết nối"):
                    st.markdown("""
                    ### Các bước khắc phục:
                    
                    1. **Kiểm tra quyền truy cập camera:**
                       - Đảm bảo browser có quyền truy cập camera
                       - Kiểm tra biểu tượng camera trên thanh địa chỉ
                       
                    2. **Thử các cài đặt khác nhau:**
                       - Giảm chất lượng video xuống "Thấp"
                       - Giảm FPS xuống 10-15
                       - Refresh trang và thử lại
                       
                    3. **Vấn đề mạng:**
                       - Thử đổi sang mạng WiFi khác
                       - Tắt VPN nếu đang sử dụng
                       - Kiểm tra tường lửa công ty/trường học
                       
                    4. **Browser issues:**
                       - Thử browser khác (Chrome, Firefox, Edge)
                       - Clear cache và cookies
                       - Tắt extensions có thể can thiệp
                       
                    5. **Nếu vẫn không được:**
                       - Sử dụng chức năng "Tải hình ảnh" thay thế
                       - Liên hệ IT support nếu ở môi trường công ty
                    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🎭 Emotion Recognition App | Powered by Lê Phụng<br>"
        "💡 Tip: Nếu gặp vấn đề kết nối, hãy thử giảm chất lượng video hoặc sử dụng chức năng tải ảnh"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

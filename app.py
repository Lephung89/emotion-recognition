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
MODEL_PATH = "best_model1.h5 (1).keras"
GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1zv4QRkSa8gKx4U3b5Ox0uL0ilUMFk1b8/view?usp=sharing"
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust':(0, 255, 255),    # Yellow
    'fear': (128, 0, 128),     # Purple
    'happy': (0, 128, 0),    # Green
    'sad': (255, 0, 0),        # Blue
    'surprise': (255, 165, 0), # Orange
    'neutral': (128, 128, 128) # Gray
}

class ModelManager:
    """Quản lý tải và cache mô hình"""
    
    @staticmethod
    @st.cache_resource
    def load_emotion_model():
        """Tải mô hình với error handling tốt hơn"""
        model_path = Path(MODEL_PATH)
        
        # Kiểm tra file tồn tại và tải trực tiếp
        if model_path.exists():
            try:
                model = load_model(str(model_path), compile=False)
                st.success(f"✅ Mô hình đã được tải thành công từ: {model_path}")
                logger.info(f"Model loaded successfully from: {model_path}")
                return model
            except Exception as e:
                st.error(f"❌ Lỗi khi tải mô hình từ file có sẵn: {e}")
                logger.error(f"Model loading error: {e}")
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
        with st.spinner("🔄 Đang tải mô hình từ Google Drive..."):
            try:
                gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
                if Path(MODEL_PATH).exists():
                    model = load_model(MODEL_PATH, compile=False)
                    st.success("✅ Tải mô hình từ Google Drive thành công!")
                    return model
                else:
                    raise FileNotFoundError("Không thể tải file từ Google Drive.")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải từ Google Drive: {e}")
                return ModelManager._manual_upload()
    
    @staticmethod
    def _manual_upload():
        """Cho phép tải mô hình thủ công"""
        st.subheader("📤 Tải mô hình thủ công")
        st.info("Vui lòng tải file mô hình (.h5 hoặc .keras) để tiếp tục.")
        
        uploaded_model = st.file_uploader(
            "Chọn file mô hình", 
            type=["h5", "keras"],
            help="File mô hình phải có định dạng .h5 hoặc .keras"
        )
        
        if uploaded_model is not None:
            try:
                with open(MODEL_PATH, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                
                model = load_model(MODEL_PATH, compile=False)
                st.success("✅ File mô hình đã được tải lên và khởi tạo thành công!")
                st.rerun()  # Refresh app
                return model
            except Exception as e:
                st.error(f"❌ Lỗi khi lưu/tải file mô hình: {e}")
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
    """Video processor được tối ưu hóa"""
    
    def __init__(self, model):
        self.model = model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Prediction caching
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_interval = 0.3  # Giảm xuống để responsive hơn
        
        # Frame skipping for better performance
        self.frame_count = 0
        self.process_every_n_frames = 3  # Xử lý mỗi 3 frame
        
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
            
            # Skip frames để tối ưu performance
            should_process = (self.frame_count % self.process_every_n_frames == 0)
            should_predict = (current_time - self.last_prediction_time) >= self.prediction_interval
            
            output_img = img.copy()
            
            if should_process:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces với parameters tối ưu
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=3, 
                    minSize=(60, 60),  # Tăng minSize để tránh false positive
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) > 0 and should_predict:
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
            
            # Vẽ kết quả
            label = f'FPS: {self.current_fps:.1f}'
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
            
            # Update session state
            if 'label' in st.session_state:
                st.session_state['label'] = label
            
            return av.VideoFrame.from_ndarray(output_img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return frame

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
    
    # Header
    st.title("🎭 Nhận Diện Cảm Xúc Khuôn Mặt")
    st.markdown("---")
    
    # Sidebar thông tin
    with st.sidebar:
        st.header("ℹ️ Thông tin ứng dụng")
        st.info("""
        **Mô hình:** InceptionV3
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
        st.write("Ứng dụng tự động tối ưu hiệu suất")
    
    # Khởi tạo session state
    if 'label' not in st.session_state:
        st.session_state['label'] = '🔄 Đang khởi tạo...'
    
    # Tải mô hình
    model = ModelManager.load_emotion_model()
    if model is None:
        st.error("❌ Không thể tiếp tục do lỗi tải mô hình.")
        st.stop()
    
    # Tabs cho các chức năng
    tab1, tab2 = st.tabs(["📷 Tải hình ảnh", "🎥 Camera trực tiếp"])
    
    with tab1:
        st.subheader("📤 Tải hình ảnh để dự đoán cảm xúc")
        st.write("Hỗ trợ định dạng: JPG, JPEG, PNG")
        
        uploaded_file = st.file_uploader(
            "Chọn hình ảnh", 
            type=["jpg", "jpeg", "png"],
            help="Hình ảnh nên chứa khuôn mặt rõ ràng để có kết quả tốt nhất"
        )
        
        if uploaded_file is not None:
            process_uploaded_image(uploaded_file, model)
    
    with tab2:
        st.subheader("🎥 Nhận diện cảm xúc từ camera")
        st.write("Cho phép truy cập camera và bắt đầu nhận diện")
        
        # Cấu hình WebRTC
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        })
        
        # WebRTC Streamer
        webrtc_ctx = webrtc_streamer(
            key="emotion-recognition",
            video_processor_factory=lambda: OptimizedVideoProcessor(model),
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 30}
                }, 
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
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🎭 Emotion Recognition App | Powered by Lê Phụng"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

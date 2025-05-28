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
import shutil

# Tải mô hình từ Google Drive hoặc cho phép tải thủ công
@st.cache_resource
def load_emotion_model():
    model_path = "best_modelnew.h5.keras"
    model = None

    # Thử tải từ Google Drive
    if not os.path.exists(model_path):
        st.info("Đang tải mô hình từ Google Drive...")
        try:
            gdown.download("https://drive.google.com/uc?id=1nB_Sr_jnm0HmMSC4ISf2bFOYPGLW15Bc", model_path, quiet=False)
            if os.path.exists(model_path):
                st.success("Tải mô hình từ Google Drive thành công!")
            else:
                raise FileNotFoundError("Không thể tải file từ Google Drive.")
        except Exception as e:
            st.error(f"Lỗi khi tải từ Google Drive: {e}")
            st.warning("Không tải được mô hình từ Google Drive. Vui lòng tải thủ công.")

    # Nếu không tải được từ Google Drive, cho phép tải thủ công
    if not os.path.exists(model_path):
        st.subheader("Tải mô hình thủ công")
        uploaded_model = st.file_uploader("Vui lòng chọn file mô hình (.h5 hoặc .keras)", type=["h5", "keras"])
        if uploaded_model is not None:
            try:
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                st.success("File mô hình đã được tải lên thành công!")
            except Exception as e:
                st.error(f"Lỗi khi lưu file mô hình: {e}")
                return None

    # Tải mô hình từ file
    if os.path.exists(model_path):
        try:
            model = load_model(model_path, compile=False)
            st.success(f"Mô hình đã được tải thành công từ: {model_path}")
            print(f"Model loaded successfully from: {model_path}")
            return model
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình: {e}")
            print(f"Model loading error: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)  # Xóa file lỗi
            return None
    st.error("Không thể tải mô hình. Vui lòng kiểm tra file hoặc thử lại.")
    return None

# Tiền xử lý ảnh khuôn mặt
def preprocess_face(face_img):
    try:
        print("Preprocessing face image with shape:", face_img.shape)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        face_img = cv2.resize(face_img, (299, 299))
        face_img = face_img.astype('float32')
        face_img = (face_img - 127.5) / 127.5
        face_img = np.expand_dims(face_img, axis=0)
        print("Preprocessed image shape:", face_img.shape)
        return face_img
    except Exception as e:
        st.error(f"Lỗi trong tiền xử lý: {e}")
        print(f"Preprocessing error: {e}")
        return None

# Dự đoán cảm xúc
def predict_emotion(face_img, model):
    try:
        if model is None:
            print("Model is None, cannot predict")
            return None, 0.0
        processed_img = preprocess_face(face_img)
        if processed_img is None:
            print("Preprocessing failed")
            return None, 0.0
        print("Predicting with processed image shape:", processed_img.shape)
        predictions = model.predict(processed_img, verbose=0)
        print("Raw predictions:", predictions)
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        print(f"Predicted: {emotion_labels[predicted_class]} ({confidence:.2f})")
        return emotion_labels[predicted_class], confidence
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
        print(f"Prediction error: {e}")
        return None, 0.0

# Class xử lý video
class VideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("Haar Cascade loaded:", not self.face_cascade.empty())
        self.last_prediction = None
        self.fps_array = []
        self.prev_frame_time = 0
        self.last_prediction_time = 0
        self.prediction_interval = 0.5

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            print("Frame shape:", img.shape)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            current_time = time.time()
            fps = 1 / (current_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
            self.prev_frame_time = current_time
            self.fps_array.append(fps)
            if len(self.fps_array) > 30:
                self.fps_array.pop(0)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.03, minNeighbors=1, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
            )
            print("Detected faces:", len(faces))
            output_img = img.copy()
            label = 'Không phát hiện khuôn mặt'
            best_emotion = None
            best_confidence = 0.0
            best_face = None

            if len(faces) > 0 and (current_time - self.last_prediction_time) >= self.prediction_interval:
                for (x, y, w, h) in faces:
                    y_offset = int(h * 0.1)
                    face_y = max(0, y - y_offset)
                    face_h = min(img.shape[0] - face_y, h + y_offset * 2)
                    face_img = img[face_y:face_y+face_h, x:x+w]
                    try:
                        emotion, confidence = predict_emotion(face_img, self.model)
                        if emotion is not None and confidence > best_confidence:
                            best_emotion = emotion
                            best_confidence = confidence
                            best_face = (x, y, w, h)
                    except Exception as e:
                        st.error(f"Lỗi khi xử lý khuôn mặt: {e}")
                        print(f"Face processing error: {e}")
                        continue
                self.last_prediction_time = current_time

            if best_emotion is not None:
                self.last_prediction = (best_emotion, best_confidence)
                x, y, w, h = best_face
                avg_fps = np.mean(self.fps_array) if self.fps_array else 0
                label = f'Cảm xúc: {best_emotion} ({best_confidence:.2f}), FPS: {avg_fps:.1f}'
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 4)
                text = f"{best_emotion}: {best_confidence:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                bg_y = max(0, y - 5)
                cv2.rectangle(output_img, (x, bg_y - text_size[1] - 4), (x + text_size[0], bg_y), (0, 0, 255), -1)
                cv2.putText(output_img, text, (x, bg_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                print("Frame drawn with label:", text)

            st.session_state['label'] = label
            print("Label updated:", label)
            return av.VideoFrame.from_ndarray(output_img, format="bgr24")
        except Exception as e:
            st.error(f"Lỗi khi xử lý video: {e}")
            print(f"Video processing error: {e}")
            return frame

# Ứng dụng Streamlit
def main():
    st.title("Nhận Diện Cảm Xúc Khuôn Mặt")
    st.write("Ứng dụng sử dụng InceptionV3 để nhận diện cảm xúc từ webcam hoặc hình ảnh tải lên.")

    if 'label' not in st.session_state:
        st.session_state['label'] = 'Đang khởi tạo...'
    if 'image_result' not in st.session_state:
        st.session_state['image_result'] = None

    # Tải mô hình
    model = load_emotion_model()
    if model is None:
        st.error("Không thể tiếp tục do lỗi tải mô hình. Vui lòng kiểm tra file hoặc tải thủ công.")
        return

    # Phần tải hình ảnh
    st.subheader("Tải hình ảnh để dự đoán cảm xúc")
    uploaded_file = st.file_uploader("Chọn một hình ảnh chứa khuôn mặt", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Hình ảnh đã tải lên", use_column_width=True)
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.03, minNeighbors=1, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) == 0:
                st.session_state['image_result'] = "Không phát hiện khuôn mặt trong hình ảnh"
            else:
                best_emotion = None
                best_confidence = 0.0
                for (x, y, w, h) in faces:
                    face_img = img_array[y:y+h, x:x+w]
                    emotion, confidence = predict_emotion(face_img, model)
                    if emotion is not None and confidence > best_confidence:
                        best_emotion = emotion
                        best_confidence = confidence
                if best_emotion is not None:
                    st.session_state['image_result'] = f"Cảm xúc: {best_emotion} ({best_confidence:.2f})"
                else:
                    st.session_state['image_result'] = "Không thể dự đoán cảm xúc"
        except Exception as e:
            st.error(f"Lỗi khi xử lý hình ảnh: {e}")
            print(f"Image processing error: {e}")
    if st.session_state['image_result']:
        st.write(st.session_state['image_result'])

    # Phần webcam
    st.subheader("Nhận diện cảm xúc từ webcam")
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]}
    )
    webrtc_streamer(
        key="emotion-recognition",
        video_processor_factory=lambda: VideoProcessor(model),
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": 320, "height": 240, "frameRate": 5}, "audio": False},
        async_processing=True,
    )
    st.write(st.session_state['label'])

if __name__ == "__main__":
    main()

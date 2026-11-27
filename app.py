import json
import os
import pickle
import threading

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

# Configuración de la página
st.set_page_config(
    page_title="Detector de Género en Tiempo Real",
    page_icon="🎥",
    layout="wide",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_genero.pkl")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "metadata_json.json")

DEFAULT_METADATA = {
    "model_type": "MobileNetV2 Transfer Learning",
    "input_shape": [224, 224, 3],
    "classes": {"0": "men", "1": "women"},
}

CLASS_LABELS_ES = {
    "men": "Hombre",
    "women": "Mujer",
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


@st.cache_resource
def load_metadata() -> dict:
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_METADATA


@st.cache_resource
def load_gender_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo en: {MODEL_PATH}"
        )
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    # El archivo .pkl contiene {'weights': [...], 'config': {...}} de Keras 3
    if isinstance(data, dict) and "config" in data and "weights" in data:
        model = tf.keras.Sequential.from_config(data["config"])
        model.set_weights(data["weights"])
        return model

    # Respaldo por si en el futuro se guarda un modelo directo
    return data


@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def preprocess_image(rgb_image: np.ndarray, target_size=(224, 224), mode: str = "mobilenet_v2") -> np.ndarray:
    resized = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32)
    if mode == "mobilenet_v2":
        # Escala de [0, 255] a [-1, 1]
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    else:
        # Escala de [0, 255] a [0, 1]
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def predict_gender(model, rgb_crop: np.ndarray, classes_map: dict, input_size=(224, 224), preprocess_mode="mobilenet_v2"):
    batch = preprocess_image(rgb_crop, target_size=input_size, mode=preprocess_mode)
    prob_class_1 = float(model.predict(batch, verbose=0)[0][0])

    if prob_class_1 >= 0.5:
        raw_label = classes_map.get("1", "women")
        confidence = prob_class_1
    else:
        raw_label = classes_map.get("0", "men")
        confidence = 1.0 - prob_class_1

    label_es = CLASS_LABELS_ES.get(raw_label, raw_label)
    return label_es, confidence, prob_class_1


class GenderVideoProcessor(VideoProcessorBase):
    def __init__(self, model, face_cascade, classes_map, input_size, preprocess_mode, detect_faces):
        self.model = model
        self.face_cascade = face_cascade
        self.classes_map = classes_map
        self.input_size = input_size
        self.preprocess_mode = preprocess_mode
        self.detect_faces = detect_faces
        self.lock = threading.Lock()
        self.latest_result = None

    def update_settings(self, preprocess_mode: str, detect_faces: bool):
        with self.lock:
            self.preprocess_mode = preprocess_mode
            self.detect_faces = detect_faces

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with self.lock:
            preprocess_mode = self.preprocess_mode
            detect_faces = self.detect_faces

        results_summary = []

        if detect_faces:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
        else:
            faces = []

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Añadir un pequeño margen alrededor del rostro
                pad_w = int(w * 0.15)
                pad_h = int(h * 0.15)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(img_rgb.shape[1], x + w + pad_w)
                y2 = min(img_rgb.shape[0], y + h + pad_h)

                face_crop = img_rgb[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                label_es, conf, prob_1 = predict_gender(
                    self.model, face_crop, self.classes_map, self.input_size, preprocess_mode
                )
                color = (255, 105, 180) if label_es == "Mujer" else (0, 191, 255)
                text = f"{label_es}: {conf * 100:.1f}%"

                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img_bgr, (x1, max(0, y1 - 30)), (x1 + 210, y1), color, -1)
                cv2.putText(
                    img_bgr,
                    text,
                    (x1 + 6, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                results_summary.append((label_es, conf, prob_1))
        else:
            # Si no se detectan rostros o el modo rostro está desactivado, evaluar el cuadro completo
            label_es, conf, prob_1 = predict_gender(
                self.model, img_rgb, self.classes_map, self.input_size, preprocess_mode
            )
            color = (255, 105, 180) if label_es == "Mujer" else (0, 191, 255)
            text = f"{label_es} ({conf * 100:.1f}%)"
            cv2.rectangle(img_bgr, (10, 10), (290, 50), (20, 20, 20), -1)
            cv2.putText(
                img_bgr,
                text,
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            results_summary.append((label_es, conf, prob_1))

        with self.lock:
            self.latest_result = results_summary

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


def main():
    st.title("🎥 Clasificador de Género en Tiempo Real")
    st.markdown(
        "Enciende tu cámara para detectar en tiempo real si la persona es **Hombre** o **Mujer** "
        "utilizando el modelo **MobileNetV2**."
    )

    try:
        metadata = load_metadata()
        model = load_gender_model()
        face_cascade = load_face_detector()
    except Exception as e:
        st.error(f"Error al cargar el modelo o recursos: {e}")
        st.stop()

    classes_map = metadata.get("classes", {"0": "men", "1": "women"})
    input_shape = metadata.get("input_shape", [224, 224, 3])
    input_size = (int(input_shape[1]), int(input_shape[0]))

    # Barra lateral con configuración e información del modelo
    with st.sidebar:
        st.header("⚙️ Configuración")
        detect_faces = st.checkbox(
            "Detectar rostro automáticamente",
            value=True,
            help="Recorta el rostro detectado antes de enviarlo al modelo. Si no detecta ninguno, evalúa toda la imagen.",
        )
        preprocess_label = st.radio(
            "Preprocesamiento de imagen",
            options=["MobileNetV2 ([-1, 1])", "Normalizado ([0, 1])"],
            index=0,
            help="Usa [-1, 1] si entrenaste con preprocess_input de MobileNetV2, o [0, 1] si usaste rescale=1./255.",
        )
        preprocess_mode = "mobilenet_v2" if "MobileNetV2" in preprocess_label else "rescale_01"

        st.divider()
        st.subheader("📊 Detalles del modelo")
        st.write(f"**Arquitectura:** {metadata.get('model_type', 'MobileNetV2')}")
        st.write(f"**Entrada:** {input_shape[0]} × {input_shape[1]} × {input_shape[2]}")
        st.write("**Clases:** `0` → Hombre (`men`), `1` → Mujer (`women`)")
        summary = metadata.get("training_summary", {})
        if summary:
            st.write(f"**Épocas:** {summary.get('epochs', '-')}")
            val_acc = summary.get("final_validation_accuracy")
            if val_acc is not None:
                st.write(f"**Precisión validación:** {val_acc * 100:.2f}%")

    tab_live, tab_photo = st.tabs(["📹 Cámara en Tiempo Real", "📸 Captura o Imagen"])

    with tab_live:
        st.info(
            "Haz clic en **START** para encender tu cámara y permitir el acceso en el navegador."
        )
        ctx = webrtc_streamer(
            key="gender-detection-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: GenderVideoProcessor(
                model=model,
                face_cascade=face_cascade,
                classes_map=classes_map,
                input_size=input_size,
                preprocess_mode=preprocess_mode,
                detect_faces=detect_faces,
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.update_settings(
                preprocess_mode=preprocess_mode,
                detect_faces=detect_faces,
            )

    with tab_photo:
        col_cam, col_upload = st.columns(2)
        with col_cam:
            cam_photo = st.camera_input("Tomar una foto")
        with col_upload:
            uploaded_file = st.file_uploader(
                "O sube una imagen", type=["jpg", "jpeg", "png", "webp"]
            )

        source_file = cam_photo if cam_photo is not None else uploaded_file
        if source_file is not None:
            image = Image.open(source_file).convert("RGB")
            img_rgb = np.array(image)

            crop_to_predict = img_rgb
            if detect_faces:
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    crop_to_predict = img_rgb[y : y + h, x : x + w]

            label_es, conf, prob_1 = predict_gender(
                model, crop_to_predict, classes_map, input_size, preprocess_mode
            )

            st.image(image, caption="Imagen analizada", width=360)
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicción", label_es)
            col2.metric("Confianza", f"{conf * 100:.2f}%")
            col3.metric("Probabilidad Mujer (clase 1)", f"{prob_1 * 100:.2f}%")


if __name__ == "__main__":
    main()

import json
import os
import pickle
import threading

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except Exception:
    cv2 = None

try:
    import av
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        WebRtcMode,
        webrtc_streamer,
    )

    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = None
except Exception as exc:
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = str(exc)

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


@st.cache_resource
def load_metadata() -> dict:
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_METADATA


def build_mobilenetv2_gender_model(input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=tuple(input_shape))
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=tuple(input_shape),
        include_top=False,
        weights=None,
    )
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


@st.cache_resource
def load_gender_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo en: {MODEL_PATH}"
        )
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "weights" in data:
        weights = data["weights"]
        # Reconstrucción directa con Functional API para evitar errores de Sequential.build en Keras 3
        model = build_mobilenetv2_gender_model((224, 224, 3))
        if len(model.weights) == len(weights):
            model.set_weights(weights)
            # Inicializar grafo de inferencia
            _ = model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
            return model

        # Respaldo con config si cambia la arquitectura
        if "config" in data:
            model = tf.keras.Sequential.from_config(data["config"])
            model.set_weights(weights)
            _ = model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
            return model

    return data


@st.cache_resource
def load_face_detector():
    if cv2 is None:
        return None
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return None
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        return None
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return None
    return detector


def preprocess_image(
    rgb_image: np.ndarray, target_size=(224, 224), mode: str = "mobilenet_v2"
) -> np.ndarray:
    pil_img = Image.fromarray(rgb_image).resize(target_size, Image.Resampling.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32)
    if mode == "mobilenet_v2":
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def predict_gender(
    model,
    rgb_crop: np.ndarray,
    classes_map: dict,
    input_size=(224, 224),
    preprocess_mode="mobilenet_v2",
):
    batch = preprocess_image(rgb_crop, target_size=input_size, mode=preprocess_mode)
    preds = model(batch, training=False)
    prob_class_1 = float(np.asarray(preds)[0][0])

    if prob_class_1 >= 0.5:
        raw_label = classes_map.get("1", "women")
        confidence = prob_class_1
    else:
        raw_label = classes_map.get("0", "men")
        confidence = 1.0 - prob_class_1

    label_es = CLASS_LABELS_ES.get(raw_label, raw_label)
    return label_es, confidence, prob_class_1


def detect_faces_rgb(face_cascade, img_rgb: np.ndarray):
    if face_cascade is None:
        return []
    gray = np.asarray(Image.fromarray(img_rgb).convert("L"))
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return faces if len(faces) > 0 else []


if WEBRTC_AVAILABLE:
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class GenderVideoProcessor(VideoProcessorBase):
        def __init__(
            self,
            model,
            face_cascade,
            classes_map,
            input_size,
            preprocess_mode,
            detect_faces,
        ):
            self.model = model
            self.face_cascade = face_cascade
            self.classes_map = classes_map
            self.input_size = input_size
            self.preprocess_mode = preprocess_mode
            self.detect_faces = detect_faces
            self.lock = threading.Lock()
            self.latest_result = None
            self.font = ImageFont.load_default()

        def update_settings(self, preprocess_mode: str, detect_faces: bool):
            with self.lock:
                self.preprocess_mode = preprocess_mode
                self.detect_faces = detect_faces

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_rgb = frame.to_ndarray(format="rgb24")

            with self.lock:
                preprocess_mode = self.preprocess_mode
                detect_faces = self.detect_faces

            faces = (
                detect_faces_rgb(self.face_cascade, img_rgb)
                if detect_faces
                else []
            )

            pil_canvas = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_canvas)
            results_summary = []

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    pad_w = int(w * 0.15)
                    pad_h = int(h * 0.15)
                    x1 = max(0, int(x) - pad_w)
                    y1 = max(0, int(y) - pad_h)
                    x2 = min(img_rgb.shape[1], int(x + w) + pad_w)
                    y2 = min(img_rgb.shape[0], int(y + h) + pad_h)

                    face_crop = img_rgb[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    label_es, conf, prob_1 = predict_gender(
                        self.model,
                        face_crop,
                        self.classes_map,
                        self.input_size,
                        preprocess_mode,
                    )
                    color = (255, 105, 180) if label_es == "Mujer" else (0, 191, 255)
                    text = f"{label_es}: {conf * 100:.1f}%"

                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    label_top = max(0, y1 - 22)
                    draw.rectangle([x1, label_top, x1 + 140, y1], fill=color)
                    draw.text(
                        (x1 + 6, label_top + 4),
                        text,
                        fill=(255, 255, 255),
                        font=self.font,
                    )
                    results_summary.append((label_es, conf, prob_1))
            else:
                label_es, conf, prob_1 = predict_gender(
                    self.model,
                    img_rgb,
                    self.classes_map,
                    self.input_size,
                    preprocess_mode,
                )
                color = (255, 105, 180) if label_es == "Mujer" else (0, 191, 255)
                text = f"{label_es} ({conf * 100:.1f}%)"
                draw.rectangle([10, 10, 210, 42], fill=(20, 20, 20))
                draw.text((18, 20), text, fill=color, font=self.font)
                results_summary.append((label_es, conf, prob_1))

            with self.lock:
                self.latest_result = results_summary

            annotated_rgb = np.asarray(pil_canvas)
            return av.VideoFrame.from_ndarray(annotated_rgb, format="rgb24")


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

    with st.sidebar:
        st.header("⚙️ Configuración")
        detect_faces = st.checkbox(
            "Detectar rostro automáticamente",
            value=face_cascade is not None,
            disabled=face_cascade is None,
            help=(
                "Recorta el rostro detectado antes de enviarlo al modelo."
                if face_cascade is not None
                else "Detector facial de OpenCV no disponible en este entorno; se analizará el cuadro completo."
            ),
        )
        preprocess_label = st.radio(
            "Preprocesamiento de imagen",
            options=["MobileNetV2 ([-1, 1])", "Normalizado ([0, 1])"],
            index=0,
            help="Usa [-1, 1] si entrenaste con preprocess_input de MobileNetV2, o [0, 1] si usaste rescale=1./255.",
        )
        preprocess_mode = (
            "mobilenet_v2" if "MobileNetV2" in preprocess_label else "rescale_01"
        )

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
        if WEBRTC_AVAILABLE:
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
        else:
            st.warning(
                "El streaming en tiempo real (`streamlit-webrtc` / `av`) no está disponible "
                f"en este entorno ({WEBRTC_IMPORT_ERROR}). "
                "Mientras tanto, puedes usar la pestaña **📸 Captura o Imagen**."
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
            img_rgb = np.asarray(image)

            crop_to_predict = img_rgb
            if detect_faces and face_cascade is not None:
                faces = detect_faces_rgb(face_cascade, img_rgb)
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


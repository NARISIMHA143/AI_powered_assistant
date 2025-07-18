import streamlit as st
import google.generativeai as genai
import pytesseract
from PIL import Image
import cv2
import numpy as np
from gtts import gTTS
import os
import uuid

# Configure Google Generative AI
genai.configure(api_key="YOUR_API_KEY")


class AudioController:
    def _init_(self):
        self.audio_file = None

    def speak_text(self, text):
        """Convert text to speech and return audio file path"""
        try:
            filename = f"audio_{uuid.uuid4().hex}.mp3"
            tts = gTTS(text)
            tts.save(filename)
            self.audio_file = filename
            return filename
        except Exception as e:
            return None

    def cleanup_audio(self):
        """Remove previously generated audio file"""
        if self.audio_file and os.path.exists(self.audio_file):
            os.remove(self.audio_file)
            self.audio_file = None


class VisualAssistanceApp:
    def _init_(self):
        self.audio_controller = AudioController()
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_scene_description(self, image):
        try:
            response = self.model.generate_content(
                [
                    "Provide a comprehensive and descriptive understanding of this scene.",
                    image,
                ]
            )
            return response.text
        except Exception as e:
            return f"Error in scene description: {str(e)}"

    def extract_text(self, image):
        try:
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            text = pytesseract.image_to_string(opencv_image)
            return text if text.strip() else "No text found in the image."
        except Exception as e:
            return f"Error in text extraction: {str(e)}"

    def detect_and_mark_objects(self, image):
        try:
            response = self.model.generate_content(
                [
                    """Analyze this image and provide object locations in the following format:
                    object1: x1,y1,x2,y2
                    where coordinates are percentages of image width/height (0-100)""",
                    image,
                ]
            )

            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            height, width = img_cv.shape[:2]
            lines = response.text.strip().split('\n')
            description = []

            for line in lines:
                if ':' in line:
                    obj, coords = line.split(':')
                    obj = obj.strip()
                    coords = coords.strip()

                    try:
                        x1, y1, x2, y2 = map(float, coords.split(','))
                        x1, x2 = int((x1 / 100) * width), int((x2 / 100) * width)
                        y1, y2 = int((y1 / 100) * height), int((y2 / 100) * height)

                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_cv, obj, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        description.append(f"Found {obj} in the image")
                    except:
                        continue

            marked_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            return marked_image, "\n".join(description)

        except Exception as e:
            return image, f"Error in object detection: {str(e)}"

    def provide_task_assistance(self, image):
        try:
            response = self.model.generate_content(
                [
                    "Analyze this image and provide context-specific guidance.",
                    image,
                ]
            )
            return response.text
        except Exception as e:
            return f"Error in task assistance: {str(e)}"


def main():
    st.set_page_config(page_title="Visual Assistance AI", page_icon=":eye:")
    st.title("Visual Assistance AI for Visually Impaired")
    st.write("Upload an image to get descriptive assistance")

    app = VisualAssistanceApp()

    if "scene_description" not in st.session_state:
        st.session_state.scene_description = ""
    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = ""
    if "object_detection" not in st.session_state:
        st.session_state.object_detection = {"image": None, "description": ""}
    if "task_guidance" not in st.session_state:
        st.session_state.task_guidance = ""

    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Scene Description"):
                st.session_state.scene_description = app.generate_scene_description(image)

        with col2:
            if st.button("Text to Speech"):
                st.session_state.extracted_text = app.extract_text(image)

        with col3:
            if st.button("Object Detection"):
                marked_image, desc = app.detect_and_mark_objects(image)
                st.session_state.object_detection = {
                    "image": marked_image,
                    "description": desc
                }

        with col4:
            if st.button("Task Assistance"):
                st.session_state.task_guidance = app.provide_task_assistance(image)

    if st.session_state.scene_description:
        st.subheader("Scene Description")
        st.write(st.session_state.scene_description)
        if st.button("Play Scene Audio"):
            audio_file = app.audio_controller.speak_text(st.session_state.scene_description)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")

    if st.session_state.extracted_text:
        st.subheader("Extracted Text")
        st.write(st.session_state.extracted_text)
        if st.button("Play Text Audio"):
            audio_file = app.audio_controller.speak_text(st.session_state.extracted_text)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")

    if st.session_state.object_detection["image"] is not None:
        st.subheader("Object Detection")
        st.image(st.session_state.object_detection["image"], use_container_width=True)
        if st.session_state.object_detection["description"]:
            st.write(st.session_state.object_detection["description"])
            if st.button("Play Object Audio"):
                audio_file = app.audio_controller.speak_text(st.session_state.object_detection["description"])
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")

    if st.session_state.task_guidance:
        st.subheader("Task Guidance")
        st.write(st.session_state.task_guidance)
        if st.button("Play Task Audio"):
            audio_file = app.audio_controller.speak_text(st.session_state.task_guidance)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")


if _name_ == "_main_":
    main()

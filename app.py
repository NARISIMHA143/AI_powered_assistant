import streamlit as st
import google.generativeai as genai
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pyttsx3
import queue
from langchain_google_genai import GoogleGenerativeAI

# Configure Google Generative AI
genai.configure(api_key="YOUR_API_KEY")


class AudioController:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.is_speaking = False

    def speak_text(self, text):
        """Play the given text as speech"""
        self.is_speaking = True
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        self.is_speaking = False

    def stop_speech(self):
        """Stop the current speech"""
        self.tts_engine.stop()
        self.is_speaking = False


class VisualAssistanceApp:
    def __init__(self):
        self.audio_controller = AudioController()
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_scene_description(self, image):
        """Generate detailed scene description using Generative AI"""
        try:
            response = self.model.generate_content(
                [
                    "Provide a comprehensive and descriptive understanding of this scene. "
                    "Include details about objects, people, colors, and overall context. "
                    "Describe the scene as if explaining it to a visually impaired person.",
                    image,
                ]
            )
            return response.text
        except Exception as e:
            return f"Error in scene description: {str(e)}"

    def extract_text(self, image):
        """Extract text from image using OCR"""
        try:
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            text = pytesseract.image_to_string(opencv_image)
            return text if text.strip() else "No text found in the image."
        except Exception as e:
            return f"Error in text extraction: {str(e)}"

    def detect_and_mark_objects(self, image):
        """Detect objects and mark them on the image"""
        try:
            # Get object detection results
            response = self.model.generate_content(
                [
                    """Analyze this image and provide object locations in the following format:
                    object1: x1,y1,x2,y2
                    object2: x1,y1,x2,y2
                    ...
                    where coordinates are percentages of image width/height (0-100)
                    Example:
                    chair: 10,20,30,40
                    table: 50,60,70,80""",
                    image
                ]
            )
            
            # Convert PIL Image to OpenCV format for processing
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            height, width = img_cv.shape[:2]
            
            # Parse response and draw boxes
            lines = response.text.strip().split('\n')
            description = []
            
            for line in lines:
                if ':' in line:
                    obj, coords = line.split(':')
                    obj = obj.strip()
                    coords = coords.strip()
                    
                    try:
                        # Convert percentage coordinates to actual pixels
                        x1, y1, x2, y2 = map(float, coords.split(','))
                        x1, x2 = int((x1/100) * width), int((x2/100) * width)
                        y1, y2 = int((y1/100) * height), int((y2/100) * height)
                        
                        # Draw rectangle
                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add label
                        cv2.putText(img_cv, obj, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        description.append(f"Found {obj} in the image")
                    except:
                        continue
            
            # Convert back to PIL Image
            marked_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            
            return marked_image, "\n".join(description)
            
        except Exception as e:
            return image, f"Error in object detection: {str(e)}"

    def provide_task_assistance(self, image):
        """Provide task-specific guidance based on image"""
        try:
            response = self.model.generate_content(
                [
                    "Analyze this image and provide context-specific guidance. "
                    "If it shows a specific task or environment, offer step-by-step advice "
                    "or helpful observations to assist the user.",
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

    # Initialize the application
    app = VisualAssistanceApp()

    # Session state for persisting results
    if "scene_description" not in st.session_state:
        st.session_state.scene_description = ""
    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = ""
    if "object_detection" not in st.session_state:
        st.session_state.object_detection = {"image": None, "description": ""}
    if "task_guidance" not in st.session_state:
        st.session_state.task_guidance = ""

    # Image upload
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Functionality Buttons
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Scene Description"):
                st.session_state.scene_description = app.generate_scene_description(image)

        with col2:
            if st.button("Text to Speech"):
                st.session_state.extracted_text = app.extract_text(image)

        with col3:
            if st.button("Object Detection"):
                marked_image, description = app.detect_and_mark_objects(image)
                st.session_state.object_detection = {
                    "image": marked_image,
                    "description": description
                }

        with col4:
            if st.button("Task Assistance"):
                st.session_state.task_guidance = app.provide_task_assistance(image)

    # Display results
    if st.session_state.scene_description:
        st.subheader("Scene Description")
        st.write(st.session_state.scene_description)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Play Scene Description Audio"):
                app.audio_controller.speak_text(st.session_state.scene_description)
        with col2:
            if st.button("Stop Scene Description Audio"):
                app.audio_controller.stop_speech()

    if st.session_state.extracted_text:
        st.subheader("Extracted Text")
        st.write(st.session_state.extracted_text)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Play Extracted Text Audio"):
                app.audio_controller.speak_text(st.session_state.extracted_text)
        with col2:
            if st.button("Stop Extracted Text Audio"):
                app.audio_controller.stop_speech()

    if st.session_state.object_detection["image"] is not None:
        st.subheader("Object Detection")
        st.image(st.session_state.object_detection["image"], 
                caption="Detected Objects", 
                use_container_width=True)
        if st.session_state.object_detection["description"]:
            st.write(st.session_state.object_detection["description"])
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Play Object Detection Audio"):
                    app.audio_controller.speak_text(st.session_state.object_detection["description"])
            with col2:
                if st.button("Stop Object Detection Audio"):
                    app.audio_controller.stop_speech()

    if st.session_state.task_guidance:
        st.subheader("Task Guidance")
        st.write(st.session_state.task_guidance)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Play Task Guidance Audio"):
                app.audio_controller.speak_text(st.session_state.task_guidance)
        with col2:
            if st.button("Stop Task Guidance Audio"):
                app.audio_controller.stop_speech()


if __name__ == "__main__":
    main()
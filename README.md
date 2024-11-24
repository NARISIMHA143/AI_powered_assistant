 Documentation for the Visual Assistance AI Application

This script implements a **Visual Assistance AI** system using **Streamlit** for the user interface and Google Generative AI, along with other libraries such as PyTesseract for OCR and OpenCV for image processing. Below is a detailed breakdown of the script:

Imports and Dependencies
1. Libraries Used:
   - `streamlit`: For creating an interactive web application.
   - `google.generativeai`: For integrating Google Generative AI.
   - `pytesseract`: For Optical Character Recognition (OCR).
   - `PIL.Image`: For image manipulation.
   - `cv2` (OpenCV): For image processing.
   - `numpy`: For handling image data arrays.
   - `pyttsx3`: For Text-to-Speech (TTS) functionality.
   - `queue`: For managing tasks.
   - `langchain_google_genai`: For advanced integration with Google Generative AI.

2. Google Generative AI Configuration:
   - API key is required to authenticate with Google Generative AI.


Classes
1. `AudioController`
   - Manages text-to-speech operations.
   - Methods:
     - `speak_text(text)`: Converts text into speech.
     - `stop_speech()`: Stops ongoing speech playback.

2. `VisualAssistanceApp`
   - Contains core functionalities for image processing and assistance.
   - Attributes:
     - `audio_controller`: Instance of `AudioController` for audio support.
     - `model`: Google Generative AI model for content generation.
   - Methods:
     - `generate_scene_description(image)`: Generates a detailed scene description using Generative AI.
     - `extract_text(image)`: Extracts text from an image using OCR.
     - `detect_and_mark_objects(image)`: Detects objects in an image, marks them with bounding boxes, and generates a description.
     - `provide_task_assistance(image)`: Provides task-specific guidance based on the image's context.



Application Logic
Streamlit Web Application (`main`)
1. Page Configuration:
   - Sets the page title and icon using `st.set_page_config`.

2. User Interaction:
   - Displays a title and description of the application.
   - Allows users to upload images for processing.

3. Session State:
   - Persists results across multiple interactions using `st.session_state`.

4. Features:
   - **Scene Description**: Generates a narrative about the uploaded image.
   - **Text Extraction**: Extracts and displays any text in the image.
   - **Object Detection**: Identifies and highlights objects in the image.
   - **Task Assistance**: Provides guidance based on the image's context.

5. Audio Features:
   - Users can play or stop audio for scene descriptions, extracted text, detected objects, or task guidance.

6. Image Display:
   - Displays the uploaded and processed images with appropriate captions.

---

Key Functions
1. Scene Description
   - Uses Generative AI to describe the image in detail, focusing on objects, people, colors, and overall context.

2. Text Extraction
   - Leverages PyTesseract to extract textual information from the image.

3. Object Detection
   - Generates bounding boxes for detected objects in the image using Generative AI and OpenCV for visualization.

4. Task Assistance
   - Provides detailed guidance based on the specific tasks depicted in the image.

---

 **How to Run**
1. Install the required dependencies:
   ```
   pip install streamlit google-generativeai pytesseract opencv-python-headless pillow pyttsx3 langchain_google_genai
   ```
2. Set up the `api_key` for Google Generative AI in the `genai.configure` function.

3. Run the script:
   ```
   streamlit run <script_name>.py
   ```
4. Open the application in a browser using the URL displayed in the terminal.




This documentation provides an overview of the script’s design, features, and functionalities to assist in its maintenance and further development.



![Screenshot 2024-11-24 120853](https://github.com/user-attachments/assets/914aa881-6552-498a-9f7b-4c5e1001aec4)

![Screenshot 2024-11-24 120913](https://github.com/user-attachments/assets/19382384-f1ea-4cf8-8ecc-8f52b019f01d)


![Screenshot 2024-11-24 120928](https://github.com/user-attachments/assets/464edb55-8250-48c5-9514-7d550bcd87fc)


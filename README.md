# Emotion Detection System

This project is a comprehensive Emotion Detection System that identifies human emotions from images, videos, live camera feeds, and audio recordings. The application is built using Flask, TensorFlow, Keras, and OpenCV.

## Project Summary

### Idea Behind Making This Project
The goal of this project is to develop an efficient and versatile emotion detection system capable of analyzing different types of media. By leveraging custom-trained models, the system aims to provide accurate emotion detection, applicable in fields such as mental health monitoring, customer service, and interactive applications.

### About the Project
The Emotion Detection System comprises two main components:
1. **Image and Video Emotion Detection**: This component processes images, videos, and live camera feeds to detect emotions using custom-trained deep learning models.
2. **Speech Emotion Detection**: This component analyzes audio recordings to determine the speaker's emotion and allows on-the-spot audio recording on the webpage.

### Software Used in Project
- Flask
- TensorFlow
- Keras
- OpenCV
- Librosa
- Bootstrap (for UI design)

### Technical Apparatus Requirements
- Python 3.7 or higher
- TensorFlow and Keras
- OpenCV
- Flask
- Librosa
- A webcam (for live camera feed)
- Microphone (for recording audio)

### Result or Working of Project
The system successfully detects emotions from images, videos, live camera feeds, and audio recordings. The processed results are displayed with the detected emotions highlighted, providing a clear and user-friendly interface.

### Research Done
The models were trained using datasets that contain various emotional expressions. Extensive testing and validation were conducted to ensure accuracy and robustness. The training process involved fine-tuning hyperparameters and using advanced techniques to improve model performance.

## Data Flow Diagram / Process Flow

### Logic & Process Flow
1. **Input**: The user provides an image, video, audio file, or uses the live camera/audio recording feature.
2. **Preprocessing**: The input data is preprocessed (e.g., resizing images, extracting features from audio).
3. **Emotion Detection**: The preprocessed data is fed into the custom-trained models.
4. **Output**: The detected emotions are displayed on the UI, with relevant visual markers or text.

## Datasets links

### Images Dataset
https://www.kaggle.com/datasets/msambare/fer2013

### Audio Dataset
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

## References

- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Documentation: https://keras.io/
- OpenCV Documentation: https://opencv.org/
- Librosa Documentation: https://librosa.org/
- Bootstrap Documentation: https://getbootstrap.com/

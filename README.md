# Plant Leaf Disease Diagnosis App 🌿

[![Streamlit](https://img.shields.io/badge/streamlit-deployed-green)](https://plant-leaf-disease-diagnosis.streamlit.app/)  
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This app lets users upload or capture images of plant leaves, and uses a deep learning model to diagnose diseases. It also gives preventive measures and possible cures/suggestions using AI-based insights.Here we have trained DCNN and CNN and saved the trained model in a .h5 file which we have later used in the flask interface which is integrated with the Quasar AI sourced from OpenRouter, which takes the predicted name of the leaf disease and adds it with the prompt which then generates a leaf disease report with its cure and prevention.
DATASET USED- PlantVillage benchmark dataset

---

## Table of Contents

- [Demo / Live App](#demo--live-app)  
- [Features](#features)  
- [Model & Architecture](#model--architecture)  
- [Tech Stack](#tech-stack)  
- [Setup / Installation](#setup--installation)  
- [Usage](#usage)  
- [Dataset](#dataset)  
- [Evaluation & Metrics](#evaluation--metrics)  
- [Future Improvements](#future-improvements)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)

---

## Demo / Live App

Try it live here:  
[Plant Leaf Disease Diagnosis](https://plant-leaf-disease-diagnosis.streamlit.app/)

> ⚠️ Note: The live app may sometimes show internal server errors due to hosting constraints or resource limits.

---

## Features

- Upload leaf images or capture via camera (webcam)  
- Classify the leaf disease among multiple classes  
- Provide disease prevention strategies & cure suggestions  
- Show confidence / probability scores  
- Display the input image back to user  
- Simple, clean UI for ease of use  

---

## Model & Architecture

- Based on convolutional neural network (CNN / DenseNet121)  
- Trained on a multiclass leaf disease dataset  
- Saved as `.h5` file (Keras)  
- Preprocessing steps include resizing, normalization, augmentation (optional)  
- Prediction pipeline wrapped inside a web app  
- Post-processing to generate suggestions via a knowledge / AI system (e.g. Quasar / Gemini AI)

---

## Tech Stack

- **Backend / Model Serving**: Flask (or FastAPI)  
- **Frontend / Dashboard**: Streamlit  
- **Deep Learning**: TensorFlow / Keras, or PyTorch (if applicable)  
- **AI Prompting / Insights**: Quasar / Gemini AI (or whichever AI system you integrated)  
- **Other libraries**: OpenCV, Pillow, numpy, pandas, etc.  
- **Hosting**: Streamlit Cloud (or other)  

---

## Setup / Installation

### Prerequisites

- Python 3.7+  
- (Optional) GPU if you want to retrain or fine-tune  

### Steps

1. Clone the repo  
   ```bash
   git clone https://github.com/mr-milannn/Plant-Leaf-Disease-Diagnosis.git
   cd Plant-Leaf-Disease-Diagnosis
   ```
   python3 -m venv venv
   source venv/bin/activate     # Linux / macOS  
   venv\Scripts\activate        # Windows

   pip install -r requirements.txt

   streamlit run app.py
## Usage

   Upload an image of a plant leaf (jpg/png) or capture via webcam

   Wait for model prediction

   View the predicted disease class, confidence score

   Read the suggestions / cure / preventive measures

   (Optional) Download the results or share

   ## Dataset

   The dataset contains images of healthy and diseased leaves (multiple classes)

   Sources: (if publicly available) e.g. “PlantVillage” or your own collected dataset

   Preprocessing includes resizing, normalization, augmentation (rotations, flips)

   ## Evaluation & Metrics

   Accuracy, precision, recall, F1-score for each class

   Confusion matrix

   ROC / AUC curves (if applicable)

   Loss & accuracy plots during training

   Include sample metrics or performance screenshots here.

   ## Future Improvements

   Add more disease classes / support more plant species

   On-device model (mobile app)

   Real-time inference on video stream

   Better UI / user experience

   More robust error handling (bad image, low lighting)

   Multilingual support

   ## Contributing

   Contributions are welcome! You can:

   Report issues or bugs

   Suggest features

   Improve UI / UX

   Add more disease classes / data

   Optimize model or inference speed

   Please fork the project, make your changes, and create a pull request.

   License

   This project is licensed under the MIT License — see the LICENSE file for details.

   ## Acknowledgments

   Thanks to the creators of the dataset (e.g. PlantVillage)

   Shoutout to libraries & frameworks (TensorFlow, Streamlit, Flask, etc.)

   Inspiration & tutorials from online communities and papers

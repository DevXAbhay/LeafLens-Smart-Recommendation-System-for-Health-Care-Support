# LEAFLENS: LEAF IDENTIFICATION SYSTEM 🌿
Dive into the world of **medicinal botany** with **LeafLens**! 🌿✨ Powered by the **Xception model**, this ML-driven tool identifies leaves and reveals their **medicinal properties**. Simply upload an image, and let LeafLens unlock nature’s healing secrets—one leaf at a time! Whether you're a **health enthusiast, herbalist, or botanist**, LeafLens is your guide to the world of medicinal plants. 🔍🍃

# ABSTRACT

This project focuses on developing an automated system for identifying various species of leaves using the Xception model, a deep learning architecture known for its efficiency in image classification tasks. The model is trained on a diverse dataset of plant leaves to enhance accuracy and reliability.

# Introduction
With the increasing interest in plant biodiversity, the need for automated identification systems has become paramount. This project aims to leverage advanced machine learning techniques to accurately classify plant leaves based on image data.

# Project Overview

Objective: To create a robust system capable of identifying and classifying leaf species from images.

Dataset: The dataset comprises images of 83 different leaf species, totaling over 7,000 images.

Model Architecture: The Xception model is utilized due to its superior performance in handling image data through depthwise separable convolutions.

# Dataset Description

The dataset consists of images sourced from various botanical collections.
Each class corresponds to a specific leaf species, facilitating supervised learning.

# Usage

Uploading an Image for Leaf Identification

The LeafLens project allows users to identify leaves by manually uploading an image. Follow these steps to get started:

Setup:

-Ensure you have Python installed on your system.

-Make sure the required libraries (streamlit, tensorflow, and numpy) are installed. You can install them using:

pip install streamlit tensorflow numpy

Run the Program:

-Navigate to the main folder of the project.

-Open the command prompt (CMD) and execute:

streamlit run main.py

-Upload an Image:

Once the Streamlit app opens in your browser, you will see an option to upload an image.
Select the image of the leaf you want to identify and upload it.

-View Prediction:

The model will process the uploaded image and display the predicted class along with confidence scores.
Additional information regarding the identified species will be shown on the screen.

-Exit the Program:

-Close the browser tab or stop the Streamlit server from the command line if needed.

# Important Notes:
-Ensure the image quality is good for better accuracy in predictions.

-The model's predictions depend on the training dataset; therefore, it may not recognize all leaf species accurately.

# Screenshot:

# Upload Image:

![Streaming](https://github.com/user-attachments/assets/6acb9e9e-bd9a-4254-aa7e-a72379437f15)

# Working on the Image:

![Working_IMG1](https://github.com/user-attachments/assets/ae889728-8c9f-4bba-ad02-4fde3659c2d0)

# Output:

![Working_IMG2](https://github.com/user-attachments/assets/eecabd23-1854-4c2e-a5d7-1f74dec5a25c)

# Conclusion:

Saving the model and its weights ensures efficient reuse for future inference or training. By integrating it into a **user-friendly interface** with image uploads, **LeafLens** becomes a powerful tool for **healthcare, herbal research, and biodiversity studies**, aiding in the exploration of medicinal plant properties. 🌿💡

# Author

ABHAY ARORA

# TanamTepat: Soil Type Detection and Crop Recommendation 🌱

This is an interactive web application built with Streamlit and TensorFlow to assist farmers and agricultural enthusiasts. By simply uploading an image of soil, the application will classify its type and provide recommendations for the most suitable crops to plant.

## ✨ Key Features

  - **Automatic Detection:** Identifies the soil type from an uploaded image (`Alluvial Soil`, `Black Soil`, `Clay Soil`, or `Red Soil`).
  - **Crop Recommendations:** Provides a list of crops that are best suited for the detected soil type.
  - **Simple User Interface:** The application is easy to use with a clean and minimalist interface.

-----

## 🚀 How It Works

The application operates in a few simple steps:

1.  **Image Upload:** The user uploads a soil image through the Streamlit interface.
2.  **Model Download:** The application automatically downloads the model file (`Model_Fix.h5`) from a GitHub URL. This file is only downloaded once when the app is first run.
3.  **Image Preprocessing:** The uploaded image is resized to **224x224** pixels and normalized to be processed by the model.
4.  **Prediction:** The preprocessed image is fed into the **TensorFlow** model to predict the soil type.
5.  **Recommendation:** Based on the prediction result, the application displays a predefined list of crops suitable for that soil type.

-----

## 🛠️ Tech Stack

  - **Web Framework:** [Streamlit](https://streamlit.io/)
  - **Machine Learning:** [TensorFlow](https://www.tensorflow.org/)
  - **Image Manipulation:** [PIL (Pillow)](https://pillow.readthedocs.io/)
  - **Data Management:** [NumPy](https://numpy.org/)
  - **HTTP Requests:** [Requests](https://requests.readthedocs.io/)

-----

## 🖥️ How to Run the Application

To run this application locally on your computer, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/netrialiarahmi/TanamTepat.git
    cd TanamTepat
    ```

2.  **Install Dependencies:**
    Make sure you have Python and `pip` installed. Create a `requirements.txt` file with the following content:

    ```
    streamlit
    tensorflow
    Pillow
    numpy
    requests
    ```

    Then, install all dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**

    ```bash
    streamlit run app.py
    ```

    The application will automatically open in your web browser.

-----

## 📚 Model & Data Notes

The model used, **`Model_Fix.h5`**, is a Deep Learning model trained for soil image classification.

-----

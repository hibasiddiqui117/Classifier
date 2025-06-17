# Classifier
Built a deep learning image classifier using TensorFlow and Keras to distinguish between various fruits and vegetables. Trained on a structured image dataset using a CNN model, then deployed a Streamlit web app for real-time predictions.

A deep learning project built using TensorFlow and Keras to classify images of fruits and vegetables. The model is trained on a custom dataset and deployed using Streamlit for real-time image prediction through a simple web interface.

 🚀 Features

* Image classification using Convolutional Neural Networks (CNN)
* Trained on fruits and vegetables dataset
* Real-time prediction via web app (Streamlit)
* Clean UI for uploading and classifying images
* Model saved and reused using `.keras` format


🛠️ Tech Stack

* Python
* TensorFlow & Keras
* Streamlit
* NumPy
* Pillow (PIL)
* Matplotlib / Plotly

📁 Folder Structure

```
Classifier/
├── app.py                       # Streamlit app
├── model/
│   └── Image_classify.keras     # Trained model
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
├── utils/
│   └── preprocess.py            # (optional) Preprocessing helpers
├── requirements.txt
└── README.md
```

 📷 Sample Usage

Launch the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser at: `http://localhost:8501`

Upload any image, and the app will classify it as a specific fruit or vegetable.

 🧪 Model Overview

* CNN with 3 Conv2D layers
* Rescaling, Dropout, and Dense layers
* Trained over 25 epochs
* Accuracy: \~90% on validation set

 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Classifier.git
   cd Fruit-Veg-Classifier
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```
 📌 To-Do

* Improve model accuracy with data augmentation
* Add image preview before prediction
* Add more fruit/vegetable categories

🙌 Acknowledgements

This project was inspired by a practical use-case for beginners learning deep learning and Streamlit for deployment.


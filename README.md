# 😴 Sleep Stage Classification  
### Multimodal Machine Learning & Deep Learning Project

Sleep Stage Classification is a Machine Learning and Deep Learning based project that automatically classifies human sleep into different stages (Wake, REM, N1, N2, N3) using physiological signals such as EEG and other derived features.  
This project helps in understanding sleep patterns and supports applications in healthcare and sleep disorder diagnosis.

---

## 📌 Problem Statement
Manual sleep stage scoring is time-consuming and prone to human error.  
This project aims to **automate sleep stage classification** using ML and DL models to achieve accurate and efficient predictions.

---

## 🧠 Sleep Stages
- **Wake**
- **N1 (Light Sleep)**
- **N2**
- **N3 (Deep Sleep)**
- **REM (Rapid Eye Movement)**

---

## ✨ Features
- Automated sleep stage prediction  
- Multimodal data fusion (signals + extracted features)  
- Deep learning models for temporal pattern learning  
- Traditional ML models for comparison  
- Visualization of hypnograms  
- Streamlit-based interactive interface  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Matplotlib  
  - Scikit-learn  
  - TensorFlow / Keras  
  - XGBoost  
- **Framework:** Streamlit  
- **Version Control:** Git & GitHub  

---

## 📂 Project Structure
```

Sleep-Stage-Classification/
│
├── app.py                  # Streamlit application
├── load_data.py             # Data loading module
├── preprocess_data.py       # Signal preprocessing
├── train.py                 # ML model training
├── train_cnn.py             # CNN model
├── train_dl.py              # Deep learning models
├── train_xgboost.py         # XGBoost classifier
├── train_best_dl.py         # Best performing DL model
├── tune.py                  # Hyperparameter tuning
├── multimodel/              # Multiple model experiments
├── generate/                # Generated outputs & plots
├── Hypnogram.png            # Sample hypnogram
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Sleep-Stage-Classification.git
cd Sleep-Stage-Classification
````

### 2️⃣ Install required packages

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
streamlit run app.py
```

---

## 🔄 Workflow

1. Data Collection
2. Signal Preprocessing
3. Feature Extraction
4. Model Training (ML & DL)
5. Model Evaluation
6. Sleep Stage Prediction
7. Visualization (Hypnogram)

---

## 📊 Output

* Classified sleep stages for each epoch
* Performance comparison of ML vs DL models
* Visual hypnogram for sleep analysis

---

## 🚀 Future Enhancements

* Real-time sleep monitoring
* Wearable device integration
* Transformer-based models
* Cloud deployment (AWS / GCP)
* Mobile application support

---

## 👩‍💻 Author

**Evani VSS Lalitha**
AI & Data Science Enthusiast

---

## 📜 License

This project is intended for educational and research purposes only.

# 🌌 Galaxy Machine Learning Project

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

## 📖 About
**Galaxy Classifier AI** is a machine learning system that classifies galaxies as **Spiral** or **Elliptical** using image features. The project features a **custom AdaBoost implementation from scratch** and an interactive Streamlit dashboard for live predictions and research insights.

## 🛠 Tech Stack
* **ML Framework:** Custom AdaBoost, scikit-learn
* **Image Processing:** OpenCV, scikit-image
* **Frontend:** Streamlit
* **Data:** Pandas, NumPy, Matplotlib, Seaborn

## ✨ Features

### 🚀 Live Prediction
* Upload galaxy images and get instant classification
* View extracted features (color stats, entropy, shape metrics)

### 📊 Research Dashboard
* **Feature Analysis:** Distribution plots, feature importance
* **Boosting Internals:** Alpha decay, error evolution visualizations
* **Model Comparison:** Benchmark against Random Forest, Gradient Boosting, SVM

### ⚙️ Technical Highlights
* **Manual AdaBoost:** Implemented from scratch with weighted decision stumps
* **Feature Extraction:** 11 features (RGB means, RGB stds, entropy, area, perimeter, circularity, eccentricity)
* **Grid Search:** Hyperparameter tuning with detailed logging

## 🚀 Quick Start

1. **Clone & Install:**
    ```bash
    git clone https://github.com/OsherBerGit/Galaxy-Machine-Learning-Project.git
    cd Galaxy-Machine-Learning-Project
    pip install -r requirements.txt
    ```

2. **Download Data:** Get `images_training_rev1.zip` and `training_solutions_rev1.csv` from [Kaggle Galaxy Zoo](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data), extract to `data/`

3. **Prepare & Train:**
    ```bash
    python src/1_prepare_data.py
    python src/2_feature_extraction.py
    python src/8_train_final_manual_model.py
    ```

4. **Run App:**
    ```bash
    streamlit run main.py
    ```

## 📁 Project Structure
```
├── main.py                   # Streamlit dashboard
├── src/
│   ├── galaxy_adaboost.py    # Manual AdaBoost implementation
│   ├── 1_prepare_data.py     # Data preprocessing
│   ├── 2_feature_extraction.py
│   └── ...                   # Analysis scripts (3-7)
├── data/                     # Dataset & results
├── models/                   # Trained models
└── plots/                    # Generated visualizations
```

---
*Data source: [Galaxy Zoo Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) on Kaggle*

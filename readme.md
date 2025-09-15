# Student Performance Prediction (Linear Regression Demo)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%20%7C%20Pandas%20%7C%20Seaborn-green.svg)

---

## 📌 Project Overview

This is a web application built with Flask that demonstrates a complete, end-to-end machine learning workflow for a college project. The application uses a **Linear Regression** model to predict a student's "Performance Index" based on key academic and lifestyle factors.

The project is designed to serve two primary user roles:
1.  **Teachers**, who can view a visual dashboard with insights derived from the student dataset.
2.  **Students**, who can submit their own data through a simple form, contributing to the dataset.

---

## ✨ Key Features

-   📊 **Interactive Teacher Dashboard**: A visual dashboard showcasing data distributions and correlations using plots like scatterplots, violin plots, and heatmaps to reveal insights.
-   📝 **Student Data Entry Form**: A simple and intuitive form for students to submit new data, which is seamlessly appended to the project's dataset.
-   🧠 **Offline Model Training**: The machine learning model is trained and saved separately, a best practice that separates the concerns of model training and application serving.
-   📁 **Clean Code Structure**: A well-organized project that separates the Flask web application, model training script, templates, and data for easy maintenance and understanding.

---

## 🤖 The Machine Learning Model

The predictive core of this project is a **Linear Regression** model trained using Scikit-learn.

-   **Target Variable**: `Performance Index` (a score from 10 to 100).
-   **Key Features**:
    -   `Hours Studied`
    -   `Previous Scores`
    -   `Extracurricular Activities` (Encoded as 1 for Yes, 0 for No)
    -   `Sleep Hours`
    -   `Sample Question Papers Practiced`
-   **Training Process**: The model is trained on 80% of the `Student_Performance.csv` dataset and evaluated on the remaining 20%. The final trained model is serialized and saved as `student_performance_model.joblib`.

---

## 📈 Dashboard & Data Insights

The teacher's dashboard is designed to provide quick, actionable insights from the student data. It includes:

-   **Correlation Heatmap**: To quickly identify the strongest positive and negative relationships between variables (e.g., the strong positive correlation between `Hours Studied` and `Performance Index`).
-   **Scatter Plots**: To visualize the direct relationship between key features like `Previous Scores` and the `Performance Index`.
-   **Violin Plots**: To understand the distribution of performance across different categories, such as students with and without extracurricular activities.

---

## 📂 Project Structure

```
.
├── app.py                      # Main Flask application
├── train_model.py              # Script to train and save the model
├── student_performance_model.joblib # The saved, pre-trained model file
├── Student_Performance.csv     # The dataset
├── requirements.txt            # Python dependencies
└── templates/
    ├── index.html              # Landing page
    ├── teacher.html            # Teacher dashboard
    └── student.html            # Student data entry form
```

---

## 🚀 Setup and Usage

### Prerequisites
-   Python 3.8 or higher
-   `pip` and `venv`

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/student-performance-predictor.git](https://github.com/your-username/student-performance-predictor.git)
cd student-performance-predictor
```

### 2. Set Up a Virtual Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate
# Or on Windows
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
This is a crucial one-time step. Run the training script from your terminal to generate the `student_performance_model.joblib` file.

```bash
python train_model.py
```

### 5. Run the Flask Application
```bash
flask run
```
Navigate to `http://127.0.0.1:5000` in your web browser to use the application.

---

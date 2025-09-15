# Student Performance Prediction (Linear Regression Demo)


[Image of students studying in a classroom]


This is a web application built with Flask that demonstrates a simple machine learning workflow for a college project. The application uses a Linear Regression model to predict a student's "Performance Index" based on several academic and lifestyle factors.

The project is divided into two main components:
1.  A **training script** that preprocesses data and trains the ML model.
2.  A **Flask web application** with a dashboard for teachers to view data insights and a form for students to submit their data.

---

## Features

-   **Teacher Dashboard**: A visual dashboard showcasing data distributions, correlations, and key relationships using various plots (scatterplot, violin plot, heatmap, etc.).
-   **Student Data Entry**: A simple form for students to contribute new data, which is appended to the dataset.
-   **Offline Model Training**: The machine learning model is trained separately for efficiency, following best practices.
-   **Clear Code Structure**: The project separates the concerns of model training and web serving.

---

## The Machine Learning Model

The core of this project is a **Linear Regression** model.

-   **Target Variable**: `Performance Index`
-   **Features**:
    -   `Hours Studied`
    -   `Previous Scores`
    -   `Extracurricular Activities` (Yes/No, encoded as 1/0)
    -   `Sleep Hours`
    -   `Sample Question Papers Practiced`
-   **Process**: The model is trained on 80% of the data and evaluated on the remaining 20% to assess its performance. The trained model is then saved to `student_performance_model.joblib`.

---

## Project Structure

```
.
├── app.py                      # Main Flask application
├── train_model.py              # Script to train and save the model
├── student_performance_model.joblib # The saved, pre-trained model file
├── Student_Performance.csv     # The dataset
├── requirements.txt            # Python dependencies
└── templates/
    ├── index.html
    ├── teacher.html
    └── student.html
```

---

## Setup and Usage

### Prerequisites
-   Python 3.8+
-   `pip` and `venv`

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <project-directory>
```

### 2. Set Up a Virtual Environment
```bash
# Create a virtual environment
python -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate
# Or on Windows
# .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
This is a crucial one-time step. Run the training script to generate the `student_performance_model.joblib` file.

```bash
python train_model.py
```

### 5. Run the Flask Application
```bash
flask run
# or
python app.py
```
Navigate to `http://127.0.0.1:5000` in your web browser to use the application.
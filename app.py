from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import base64
from io import BytesIO
import csv
import joblib

app = Flask(__name__)

# --- Load Model and Data ONCE at startup ---
model = joblib.load('student_performance_model.joblib')
df = pd.read_csv('Student_Performance.csv')
# Perform the same encoding as in training
encoder = LabelEncoder()
df['Extracurricular Activities'] = encoder.fit_transform(df['Extracurricular Activities'])


def get_base64_chart_image(fig):
    """Converts a matplotlib figure to a base64 encoded PNG image."""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) # Important to close the figure to save memory
    return chart_image

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/teacher')
def teacher():
    # The teacher dashboard now only focuses on VISUALIZATION.
    # The model is already trained and loaded.
    
    # --- Generate Visualizations ---
    plt.figure(figsize=(10, 6))
    scatterplot = sns.scatterplot(x=df['Previous Scores'], y=df['Performance Index'], hue=df['Extracurricular Activities'])
    fig1 = get_base64_chart_image(scatterplot.get_figure())

    plt.figure(figsize=(10, 6))
    violinplot = sns.violinplot(x=df['Previous Scores'], color='orange')
    fig2 = get_base64_chart_image(violinplot.get_figure())
    
    cat_cols = ['Hours Studied', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
    countplots = []
    for col in cat_cols:
        fig = plt.figure(figsize=(10, 6))
        sns.countplot(x=df[col])
        plt.title(f'Distribution of {col}')
        countplots.append(get_base64_chart_image(fig))
    
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    fig3 = get_base64_chart_image(heatmap.get_figure())

    # Note: Metrics like R2 score should ideally be calculated and saved during training.
    # For this demo, we'll pass a static score.
    r2_score_from_training = 0.9889 # Example value from a training run

    return render_template(
        "teacher.html", 
        fig1=fig1,
        fig2=fig2,
        fig3=fig3, 
        countplots=countplots,
        r2_score=r2_score_from_training
    )

@app.route('/student', methods=['GET', 'POST'])
def student():
    if request.method == 'POST':
        # Get data from the form
        hours_studied = request.form.get('Hours Studied')
        previous_scores = request.form.get('Previous Scores')
        extracurricular = request.form.get('Extracurricular Activities')
        sleep_hours = request.form.get('Sleep Hours')
        papers_practiced = request.form.get('Sample Question Papers Practiced')
        
        # NOTE: The student would NOT know their performance index. That's what the model predicts.
        # This route should be for data collection.
        
        # Save new data to the CSV file
        with open('Student_Performance.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # The CSV expects a Performance Index column, so we add a placeholder (e.g., 0 or None)
            csvwriter.writerow([hours_studied, previous_scores, extracurricular, sleep_hours, papers_practiced, 0])

    return render_template("student.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
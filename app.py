from flask import Flask, render_template, request
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import base64
from io import BytesIO
import csv
import os
import warnings
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


for dirname, _, filenames in os.walk('C:/Users/Sakhare Sohan/Desktop/PrachiMaam'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('Student_Performance.csv')
df.isna().sum()
df.drop_duplicates(inplace=True)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Extracurricular Activities'] = encoder.fit_transform(df['Extracurricular Activities'])

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

app = Flask(__name__)

@app.route('/', methods=(['GET']))
def index():

    return render_template("index.html")


@app.route('/teacher', methods=['GET', 'POST'])
def teacher():

    from sklearn.model_selection import train_test_split
    x = df.drop('Performance Index', axis=1)
    y = df['Performance Index']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create an imputer instance with strategy (e.g., 'mean', 'median', 'most_frequent')
    imputer = SimpleImputer(strategy='mean')

    # Fit and transform the imputer on x_train to fill missing values
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)
    y_train_imputed = imputer.fit_transform(y_train.values.reshape(-1, 1))
    y_test_imputed = imputer.transform(y_test.values.reshape(-1, 1))


    # Create and fit the LinearRegression model
    model = LinearRegression()
    model.fit(x_train_imputed, y_train_imputed)

    pred = model.predict(x_test_imputed)

    train_pred = model.predict(x_train_imputed)

    print("R2 Score on test data : ", r2_score(y_test_imputed, pred))

    print("R2 Score on train data : ", r2_score(y_train_imputed, train_pred))

    adjusted_r2_test = 1 - (1 - 0.98) * (2469 - 5) / (2469 - 5 - 1)

    adjusted_r2_train = 1 - (1 - 0.98) * (7404 - 5) / (7404 - 5 - 1)

    print("Adjusted r2 score for test data : ", (adjusted_r2_test))
    print("Adjusted r2 score for train data : ", (adjusted_r2_train))

    mean_square_test = mean_squared_error(y_test_imputed, pred)
    mean_square_train = mean_squared_error(y_train_imputed, train_pred)
    print("mean_squared_error for test data : ", mean_squared_error(y_test_imputed, pred))
    print("mean_squared_error for train data : ", mean_squared_error(y_train_imputed, train_pred))

    
    # Evaluate the model
    score = model.score(x_test_imputed, y_test_imputed)
    print(f'R-squared score: {score}')

    scatterplot = sns.scatterplot(x=df['Previous Scores'], y=df['Performance Index'], hue=df['Extracurricular Activities'])
    fig1 = get_base64_chart_image(scatterplot)
    violinplot = sns.violinplot(df['Previous Scores'], orient='h',color='orange')
    fig2 = get_base64_chart_image(violinplot)
    
    cat_col = ['Hours Studied', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
    countplots = []
    for i in cat_col:
        countplot = plt.figure(figsize=(10, 6))
        sns.countplot(x=df[i], hue=df[i])
        countplot_image = get_base64_chart_image(countplot)
        countplots.append(countplot_image)
        plt.close(countplot)
    
    heatmap = sns.heatmap(df.drop('Performance Index', axis=1).corr(), annot=True)
    fig3 = get_base64_chart_image(heatmap)

    return render_template("teacher.html", fig3 = fig3, fig2 = fig2, countplots = countplots, violinplot = violinplot, fig1 = fig1, adjusted_r2_test = adjusted_r2_test, adjusted_r2_train = adjusted_r2_train, mean_square_test = mean_square_test, mean_square_train = mean_square_train)

@app.route('/student', methods=['GET', 'POST'])
def student():
    
    if request.method == 'POST':
        HoursStudied = request.form.get('Hours Studied')
        PreousScores = request.form.get('Previous Scores')
        ExtracurricularActivities = request.form.get('Extracurricular Activities')
        SleepHours = request.form.get('Sleep Hours')
        PaperPratice = request.form.get('Sample Question Papers Practiced')
        PerformanceIndex = request.form.get('Performance Index')
        
        # Save data to a CSV file
        with open('Student_Performance.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([HoursStudied, PreousScores, ExtracurricularActivities, SleepHours, PaperPratice, PerformanceIndex])

    return render_template("student.html")


def get_base64_chart_image(fig):
    # Convert chart to base64-encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return chart_image

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
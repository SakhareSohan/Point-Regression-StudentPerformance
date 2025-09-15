import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Load and Preprocess Data
df = pd.read_csv('Student_Performance.csv')
df.drop_duplicates(inplace=True)

# Encode categorical features
encoder = LabelEncoder()
df['Extracurricular Activities'] = encoder.fit_transform(df['Extracurricular Activities'])

# 2. Define Features (X) and Target (y)
x = df.drop('Performance Index', axis=1)
y = df['Performance Index']

# 3. Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4. Handle Missing Values (Imputation)
# Note: You only need to impute X. Imputing y is generally not recommended.
imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)

# 5. Train the Model
model = LinearRegression()
model.fit(x_train_imputed, y_train)

# 6. Evaluate and Print Metrics
y_pred = model.predict(x_test_imputed)
score = r2_score(y_test, y_pred)
print(f"Model training complete.")
print(f"R-squared score on test data: {score:.4f}")

# 7. Save the Trained Model
joblib.dump(model, 'student_performance_model.joblib')
print("Model saved as 'student_performance_model.joblib'")
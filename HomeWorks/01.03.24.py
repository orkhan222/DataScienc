import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv(r'C:\Users\user\Desktop\DataAnalyst\Datasets\churn.csv')
df.head(2)

df.describe()

df.columns

selected_features = ['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary']

X = df[selected_features]
y = df[['Exited']]


X["Geography"].value_counts()

encoding = OneHotEncoder()
t = encoding.fit_transform(X[["Geography", "Gender"]])
X[["France", "Germany", "Spain", "Female", "Male"]] = pd.DataFrame(t.todense())

X.drop(columns=["Geography", "Gender"], inplace=True)

churn_model = DecisionTreeClassifier(random_state = 0)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0, train_size=0.8)

churn_model.fit(train_X, train_y)

val_prediction = churn_model.predict(val_X)

accuracy_score(val_y, val_prediction)

churn_model = DecisionTreeClassifier(random_state=0)
joblib.dump(churn_model, 'churn_model.joblib')

app = Flask(__name__)

model = joblib.load('churn_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_for_prediction = pd.DataFrame([data])
    prediction = model.predict(data_for_prediction)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000, debug=True)


model = joblib.load('churn_model.joblib')

data_for_prediction = pd.DataFrame({
    "CreditScore": [600],
    "Age": [40],
    "Tenure": [3],
    "Balance": [60000],
    "NumOfProducts": [2],
    "HasCrCard": [1],
    "IsActiveMember": [1],
    "EstimatedSalary": [50000],
    "Geography_France": [1],
    "Geography_Germany": [0],
    "Geography_Spain": [0],
    "Gender_Female": [0],
    "Gender_Male": [1]
})

predictions = model.predict(data_for_prediction)

print(predictions)

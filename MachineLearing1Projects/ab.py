# import
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import pandas as pd
import joblib

df = pd.read_csv(r'C:\Users\user\Desktop\Code\Datasets\churn.csv')


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


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Serialize the model
joblib.dump(model, 'churn_model.pkl')


app = Flask(__name__)

# Load the serialized model
model = joblib.load('churn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    # Assuming preprocessing is applied to the input data as necessary
    prediction = model.predict(df)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
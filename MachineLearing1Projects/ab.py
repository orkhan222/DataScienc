# import
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'C:\Users\user\Desktop\DataAnalyst\Datasets\train.csv')


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
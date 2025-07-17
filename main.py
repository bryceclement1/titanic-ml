import pandas as pd

from sklearn.model_selection import train_test_split

train_df = pd.read_csv("data/train.csv")
print(train_df.head())

# Drop columns we won't use (Cabin and Ticket for now)
train_df = train_df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

# Fill missing Age with median
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# Fill missing Embarked with mode
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


X = train_df.drop('Survived', axis=1) #data w/o labels
y = train_df['Survived'] # labels

#split data and labels 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#train model with LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

#train model with RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

#Check Feature Importance
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.show()


# Load test data 
test_df = pd.read_csv("data/test.csv")
true_labels = pd.read_csv("data/results.csv")  # This contains actual Survived values

# Save PassengerId for merging
passenger_ids = test_df['PassengerId']

# Preprocess test data
test_df = test_df.drop(['Cabin', 'Ticket', 'Name'], axis=1)
test_df['Age'] = test_df['Age'].fillna(train_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].median())
test_df['Embarked'] = test_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_df = test_df.drop(['PassengerId'], axis=1)

# Predict
predictions = model.predict(test_df)

# Create prediction DataFrame 
prediction_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

# Sort and compare with true labels
prediction_df = prediction_df.sort_values('PassengerId').reset_index(drop=True)
true_labels = true_labels.sort_values('PassengerId').reset_index(drop=True)

prediction_df.to_csv("predictions.csv", index=False) #outpu csv for predictions

#accuracy score for predictions
accuracy = accuracy_score(true_labels['Survived'], prediction_df['Survived'])
print("✅ Accuracy on Kaggle test data:", accuracy)
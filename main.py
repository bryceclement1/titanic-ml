import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

############
# Training the model using train.csv data
############

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

############
# Investivating importance of the features
############

sns.set(style="whitegrid")

# Feature importance
importances = model.feature_importances_
features = X.columns
feat_imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)


# Create DataFrame to sort feature importances
feat_imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)


# Create bins for Age and Fare
train_df['AgeBin'] = pd.cut(train_df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 80], right=False)
train_df['FareBin'] = pd.cut(train_df['Fare'], bins=[0, 10, 20, 30, 50, 100, 600], right=False)

# Grouped stats for plots
age_grouped = train_df.groupby('AgeBin', observed=False).agg(
    Passengers=('Survived', 'count'),
    SurvivalRate=('Survived', 'mean')
).reset_index()

fare_grouped = train_df.groupby('FareBin', observed=False).agg(
    Passengers=('Survived', 'count'),
    SurvivalRate=('Survived', 'mean')
).reset_index()

age_grouped['AgeBin'] = age_grouped['AgeBin'].astype(str)
fare_grouped['FareBin'] = fare_grouped['FareBin'].astype(str)

# 2x2 Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Feature Importance (Top-Left) 
sns.barplot(x="Importance", y="Feature", data=feat_imp_df, ax=axes[0, 0])
axes[0, 0].set_title("Feature Importance (Random Forest)")
axes[0, 0].set_xlabel("Importance")
axes[0, 0].set_ylabel("Feature")

# Age vs Survival (Top-Right)
ax = axes[0, 1]
sns.barplot(data=age_grouped, x="AgeBin", y="Passengers", ax=ax, color='lightgray')
ax2 = ax.twinx()
sns.lineplot(data=age_grouped, x="AgeBin", y="SurvivalRate", ax=ax2, color='blue', marker='o', label='Survival Rate')
ax.set_title("Age Bins: Passengers & Survival Rate")
ax.set_ylabel("Number of Passengers")
ax2.set_ylabel("Survival Rate")
ax.set_xlabel("Age Group")
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right')

# Fare vs Survival (Bottom-Left)
ax = axes[1, 0]
sns.barplot(data=fare_grouped, x="FareBin", y="Passengers", ax=ax, color='lightgray')
ax2 = ax.twinx()
sns.lineplot(data=fare_grouped, x="FareBin", y="SurvivalRate", ax=ax2, color='green', marker='o', label='Survival Rate')
ax.set_title("Fare Bins: Passengers & Survival Rate")
ax.set_ylabel("Number of Passengers")
ax2.set_ylabel("Survival Rate")
ax.set_xlabel("Fare Range")
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right')

# Sex vs Survival (Bottom-Right)
sns.barplot(data=train_df, x="Sex", y="Survived", ax=axes[1, 1])
axes[1, 1].set_title("Survival Rate by Sex")
axes[1, 1].set_xlabel("Sex")
axes[1, 1].set_ylabel("Survival Rate")
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(["Male", "Female"])

# Final layout
plt.tight_layout()
plt.show()

############
# Run model on test.csv data
############

# Load test data 
test_df = pd.read_csv("data/test.csv")
true_labels = pd.read_csv("data/test_results.csv")  # This contains actual Survived values

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

#output csv for predictions
prediction_df.to_csv("predictions.csv", index=False) 

#accuracy score for predictions
accuracy = accuracy_score(true_labels['Survived'], prediction_df['Survived'])
print("✅ Accuracy on Kaggle test data:", accuracy)
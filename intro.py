import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

warnings.filterwarnings('ignore')

# Creating a synthetic dataset
data = {
    'CustomerID': range(1, 101),
    'Age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65] * 10,
    'MonthlyCharge': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140] * 10,
    'CustomerServiceCalls': [1, 2, 3, 4, 0, 1, 2, 3, 4, 0] * 10,
    'Churn': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'] * 10
}
df = pd.DataFrame(data)

# Splitting the dataset into features and target variable
X = df[['Age', 'MonthlyCharge', 'CustomerServiceCalls']]
y = df['Churn']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Making predictions on the test set
y_pred = clf.predict(X_test)

# Evaluating the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Visualizing the decision tree
plt.figure(figsize=(10, 6))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Churn', 'Churn'])
plt.title('Decision Tree for Predicting Customer Churn')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")

print("Basic Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
sns.set(style="whitegrid")
plt.figure(figsize=(12, 10))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 2, i+1)
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='Survived', y=col, data=df)
    plt.title(f'{col} by Survival')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
corr = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')

plt.subplot(1, 3, 2)
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Pclass vs Survival')

plt.subplot(1, 3, 3)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Sex vs Survival')

plt.tight_layout()
plt.show()

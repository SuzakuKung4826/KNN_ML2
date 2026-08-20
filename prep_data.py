import pandas as pd

df = pd.read_csv('/tmp/data/titanic.csv')

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

clean = df[['Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Survived']].copy()
clean['Age'] = clean['Age'].round(1)
clean['Fare'] = clean['Fare'].round(2)

clean.to_csv('/tmp/work/KNN_ML2/Titanic.csv', index=False)
print(clean.shape)
print(clean.head())
print(clean.describe())
print(clean['Survived'].value_counts())

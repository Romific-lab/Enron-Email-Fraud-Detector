import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier 
import joblib

#Get data
df = pd.read_csv("fraud_email_.csv")
print(df.head())

#Clean data
df = df.dropna()
df = df.drop_duplicates()

print("Shape: ", df.shape)
print("Length: ", len(df))

#Slight class imbalance but that's ok (Can be fixed with SMOTE?)
#sns.countplot(data=df,x=df['Class'])
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(df.Text, df.Class, test_size = 0.25)

clf = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
joblib.dump(clf, 'fraud_detector.joblib')
import joblib
import os
import pandas as pd

clf = joblib.load('fraud_detector.joblib')
fraud_email_paths = []

df = pd.read_csv("emails.csv")
print(df.head())

#Clean data
df = df.dropna()
df = df.drop_duplicates()

length = len(df)

fraud_emails = []

for index, row in df.iterrows():
    print(f"Processing email {index + 1} of {length} — {length - index - 1} remaining")
    if clf.predict([row['message']]) == 1:
        fraud_emails.append(row['message'])

fraud_df = pd.DataFrame(fraud_emails, columns=["message"])
fraud_df.to_csv("fraud_predictions.csv", index=False)
print(f"Detected {len(fraud_emails)} fraudulent emails.")
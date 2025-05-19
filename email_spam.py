
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('path\\to\email_origin\\email_origin.csv')
dff = pd.read_csv('path\\to\\email_text\\email_text.csv')


tfidf_vectorizer = TfidfVectorizer()
label_encoder = LabelEncoder()
df['origin'].fillna('', inplace=True)
df['label'] = label_encoder.fit_transform(df['label'])  # e.g., Positive=2, Negative=0, Neutral=1
df['origin'] = df['origin'].fillna('')

X = tfidf_vectorizer.fit_transform(df['origin'])
Y = label_encoder.fit_transform(df['label'])

X_test = tfidf_vectorizer.transform(dff['text'])
Y_test = label_encoder.transform(dff['label'])


model = MultinomialNB()

model.fit(X, Y)

y_predict = model.predict(X_test)

accuracy = accuracy_score(Y_test, y_predict)
print(f"\nAccuracy Rate: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_predict))

print(f"\nAccuracy: {accuracy_score(Y_test, y_predict):.2f}")

#joblib.dump(model, 'C:/Users/AnanthM/Documents/New folder/model.joblib')
#joblib.dump(tfidf_vectorizer, 'C:/Users/AnanthM/Documents/New folder/tfidf_vectorizer.joblib')
#joblib.dump(label_encoder, 'C:/Users/AnanthM/Documents/New folder/label_encoder.joblib')




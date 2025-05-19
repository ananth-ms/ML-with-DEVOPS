from fastapi import FastAPI
import uvicorn
import joblib
import numpy as np
#from email_spam import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer


model = joblib.load('path\\to\\model.joblib')
label_encoder = joblib.load('path\\to\\label_encoder.joblib')
tfidf_vectorizer = joblib.load('path\\to\\tfidf_vectorizer.joblib')

#tfidf_vectorizer = TfidfVectorizer()

app = FastAPI()

@app.get("/")

def read_root():
    return {"mesage": "Email Spam Detection API"}

@app.post("/predict/")
def predict_spam(msg: str):

    email_vector = tfidf_vectorizer.transform([msg])

    prediction = model.predict(email_vector)

    label = "Spam" if prediction[0] == 1 else "Not Spam"

    return {"prediction": label}

msg = "All space usage is normal, and there are no slow-running queries. The database is also functioning normally"
prediction = predict_spam(msg)
print(prediction)

if __name__ == "__main__":
    uvicorn.run(app)

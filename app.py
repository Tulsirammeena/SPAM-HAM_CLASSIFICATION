import pickle
import streamlit as st
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y =[]
    sub = ['subject']
    for i in text:
      if i.isalnum():
        y.append(i)
    text = y[:]
    y.clear()
    for i in text:
      if i not in stopwords.words('english') and i not in string.punctuation and i not in sub:
        y.append(i)
    text = y[:]
    y.clear()

    for i in text:
      y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.title("Email Spam Classifier")
input_email=st.text_area("Enter the Email")
if st.button('Predict'):

    #1. preprocess
    transformed_mail = transform_text(input_email)
    #2. vectorize
    vector_input = tfidf.transform([transformed_mail])
    #3. predict
    result = model.predict(vector_input)[0]
    #4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


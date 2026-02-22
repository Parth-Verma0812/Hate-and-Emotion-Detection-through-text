import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# nltk.download('stopwords')
# nltk.download('punkt_tab')

# python -m nltk.downloader stopwords punkt using this line
with open('tokenizer.json') as f:
    data=f.read()

tokenizer=tokenizer_from_json(data)
model=load_model('multi_task-emotion-hate.keras',compile=False)

def remove_stopwords(row):
  new_row=[]
  row=word_tokenize(row)
  row=[word.lower() for word in row]
  for word in row:
    if word in stopwords.words('english'):
      new_row.append('')
    else:
      new_row.append(word)
  return " ".join(new_row)

def clean_text(text):
  text=text.lower()
  row=word_tokenize(text)
  row=[word for word in row if word.isalpha()==True]
  return " ".join(row)



def pipeline(text):
  text=clean_text(text)
  text=remove_stopwords(text)

  tokenized_text=tokenizer.texts_to_sequences([text])
  padded_text=pad_sequences(tokenized_text,maxlen=637,padding='post')

  preds=model.predict({'Emotion_Input_Layer':padded_text,'Hate_Input_Layer':padded_text})

  emo_preds=np.argmax(preds[0],axis=1)[0]
  hate_preds=np.argmax(preds[1],axis=1)[0]

  emotion_labels = ['sadness','joy','love','anger','fear','surprise']
  hate_labels = ['Offensive Speech','Neither','Hate Speech']

  emotion=emotion_labels[emo_preds]
  hate=hate_labels[hate_preds]

  return emotion,hate


st.set_page_config(page_title='Emotion and Hate Speech Detector')
st.title('Emotion and Hate Speech Detection')
st.write('Enter text below to detect emotion and hate category')
user_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if user_input.strip() != "":
        emotion, hate = pipeline(user_input)

        st.success("Analysis Complete!")

        st.subheader("Results")
        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Hate Category:** {hate}")

    else:
        st.warning("Please enter some text.")
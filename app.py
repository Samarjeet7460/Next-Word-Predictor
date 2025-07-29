from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import numpy as np
import streamlit as st
import pickle

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('next_word.h5')

st.title('Next Word Predictor')
st.write('The next word prediction is based on "The Project Gutenberg eBook of Pride and Prejudice"')

text = st.text_input(label='Enter text')

if st.button('Predict'):
    if not text:
        st.error('Input cannot be empty!')
    else:
        for i in range(10):
            token_text = tokenizer.texts_to_sequences([text])[0]
            padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
            pos = np.argmax(model.predict(padded_token_text))
            for word, index in tokenizer.word_index.items():
                if index == pos:
                    text = text + " " + word
                    st.success(text)
                    time.sleep(2)

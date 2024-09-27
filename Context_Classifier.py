import pyaudio
import numpy as np
import librosa
import tensorflow as tf
import wave
import openai
import time
import re
from transformers import pipeline
import joblib
from tensorflow.keras.models import load_model
from connection import Connection
import qi
from playsound import playsound
import os

#creating the connection
pepper = Connection()
session = pepper.connect('10.0.0.244', '9559')

# Create a proxy to the AL services
behavior_mng_service = session.service("ALBehaviorManager")
tts_service = session.service("ALTextToSpeech")

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load a pre-trained model for sentiment analysis
nlp = pipeline("sentiment-analysis")
openai.api_key =  os.getenv('OPENAI_API_KEY') 

# Load the CNN model for ambient sound detection
# model = load_model(r"C:\Users\s448160\OneDrive - University of Canberra - STAFF\PhD\Studies\HRI\CIMFSMS\3 Classes\Ambient Sound\emergency_model.h5")
model = load_model(r"emergency_model.h5")
input_shape = model.input_shape[1:]  # Exclude the batch dimension

# Load the Naive Bayes model for final emergency classification
# nb_model = joblib.load(r"C:\Users\s448160\OneDrive - University of Canberra - STAFF\PhD\Studies\HRI\CIMFSMS\3 Classes\NB\NB_model.joblib")
nb_model = joblib.load(r"NB_model.joblib")

# Play an animation
def animation(state,prompt):

    behavior_mng_service .stopAllBehaviors()
#    behavior_mng_service .startBehavior("pkg/"+state)
    tts_service.say(prompt)
    
# Function to extract MFCCs from real-time audio for ambient sound classification
def preprocess_audio(audio, sr, n_mfcc=40, n_fft=2048, hop_length=512, fixed_length=200):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Pad or truncaipte the MFCC features to fixed_length
    if mfccs.shape[1] < fixed_length:
        pad_width = fixed_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :fixed_length]
    
    return mfccs

# Function to classify audio (ambient sound)
def classify_real_time_audio(model, input_shape, sr=16000):
    p = pyaudio.PyAudio()
    chunk_size = 1024
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sr,
                    input=True,
                    frames_per_buffer=chunk_size)

    frames = []
    for i in range(0, int(sr / chunk_size * 3)):  # 3 seconds
        data = stream.read(chunk_size)
        frames.append(np.frombuffer(data, dtype=np.float32))

    audio = np.concatenate(frames, axis=0)
    mfccs = preprocess_audio(audio, sr=sr, fixed_length=input_shape[1])
    mfccs = mfccs.reshape(1, *input_shape)

    prediction = model.predict(mfccs)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    class_labels = ['Alarmed', 'Social', 'Disengaged']

    #print(predicted_class, class_labels[predicted_class], confidence)

    if class_labels[predicted_class] == 'Alarmed':
        confidence_ambient = confidence*0.3
    elif class_labels[predicted_class] == 'Social':
        confidence_ambient = confidence*0.7
    else: 
        confidence_ambient = confidence*0.2
        

    # Save recorded audio as mic.wav
    with wave.open("mic.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
        wf.setframerate(sr)
        wf.writeframes(b''.join(frames))

    return predicted_class, confidence_ambient, class_labels[predicted_class]

# Function to process speech-to-text transcription and analyze sentiment
def process_speech_to_text_and_sentiment():
    file = open("mic.wav", "rb")
    transcription = openai.Audio.transcribe("whisper-1", file)
    transcription = transcription["text"]
    print(transcription)
    # Filter transcription to only allow English text (removes non-ASCII characters, emojis, etc.)
    transcription = re.sub(r'[^a-zA-Z\s]', '', transcription).strip()

    if len(transcription) == 0:
        return 2, 0, 0, "Disengaged"

    sentiment_result = nlp(transcription)[0]
    sentiment_label = sentiment_result['label']
    sentiment_score = sentiment_result['score']

    alarmed_keywords = ['emergency', 'help', 'accident', 'fire', 'danger']
    found_keywords = any(word in transcription.lower() for word in alarmed_keywords)

    if (sentiment_label == 'NEGATIVE' and sentiment_score > 0.7) and found_keywords:
        return 0, max(sentiment_score, 0.7), 1, "Alarmed"
    
    social_keywords = ['hi', 'hello', 'hey', 'how are you']
    identified = any(word in transcription.lower() for word in social_keywords)
    
    if identified and (sentiment_label == 'NEGATIVE'):
        return 1, sentiment_score, 0.4, "Social"
    if not identified and (sentiment_label == 'NEGATIVE'):
        return 1, sentiment_score, 0.35, "Social"
    if identified and (sentiment_label == 'POSITIVE'):
        return 1, 1-sentiment_score, 0.4, "Social"
    if not identified and (sentiment_label == 'POSITIVE'):
        return 1, 1 - sentiment_score, 0.35, "Social"

# Real-time listener and processor
def listen_and_process():
    sr = 16000  # Sample rate
    while True:
        # Step 1: Record and classify ambient sound
        ambient_class, ambient_conf, ambient_label = classify_real_time_audio(model, input_shape, sr=sr)
        
        # Step 2: Process speech-to-text and sentiment analysis
        speech_class, sentiment_conf, keyword_conf, speech_label = process_speech_to_text_and_sentiment()

        # Step 3: Combine results using Naive Bayes
        context_label, final_label = classify_context(ambient_conf, keyword_conf, sentiment_conf)

        # Display the results
        print(f"Ambient: {ambient_label} (Conf: {ambient_conf:.2f}), Speech: {speech_label} (Keyword Conf: {keyword_conf:.2f}) (Sentiment Conf: {sentiment_conf: .2f})")
    #    print(f"Final Context: {context_label}\n")
        print(f"Final Context: {final_label}\n")
        time.sleep(3)  # Wait for 3 seconds before repeating

# Function to combine the two models using Naive Bayes for context classification and print confidence
def classify_context(ambient_confidence, keyword_confidence, sentiment_confidence):
    X = np.array([[ambient_confidence, keyword_confidence, sentiment_confidence]])
    
    # Get the predicted class and the corresponding probability for each class
    combined_class = nb_model.predict(X)[0]
    class_probs = nb_model.predict_proba(X)[0]
    
    class_labels = ['Alarmed', 'Social', 'Disengaged']
    context_label = class_labels[combined_class]
    
    # Print the probabilities for each class
    print(f"Naive Bayes Confidence: {dict(zip(class_labels, class_probs))}")
    final_label = ['Alarmed', 'Alert', 'Social', 'Passive', 'Disengaged']
    prob = class_probs[combined_class]
    if context_label == 'Disengaged' and 0.8 < prob < 1:
        final_label = 'Disengaged'
        animation(0,"Disengaged")
    elif context_label == 'Disengaged' and 0 < prob < 0.7:
        final_label = 'Passive'
        animation(0,"Passive")
    elif context_label == 'Social' and 0.8 < prob < 1:
        final_label = 'Social'
        animation(0,"Social")
    elif context_label == 'Social' and 0 < prob < 0.7:
        final_label = 'Passive'
        animation(0,"Passive")
    elif context_label == 'Alert' and 0.8 < prob < 1:
        final_label = 'Alarmed'
        animation(0,"Alarmed")
    elif context_label == 'Alert' and 0 < prob < 0.7:
        final_label = 'Alert'
        animation(0,"Alert")
    
    return context_label, final_label

if __name__ == "__main__":
    listen_and_process()

import pyaudio
import numpy as np
import wave
import openai
import time
import re

openai.api_key = 'key'

class Recorder:
    def __init__(self, 
                 wavfile, 
                 chunksize=8192, 
                 dataformat=pyaudio.paInt16, 
                 channels=2, 
                 rate=44100):
        self.filename = wavfile
        self.chunksize = chunksize
        self.dataformat = dataformat
        self.channels = channels
        self.rate = rate
        self.recording = False
        self.pa = pyaudio.PyAudio()

    def start(self):
        self.wf = wave.open(self.filename, 'wb')
        self.wf.setnchannels(self.channels)
        self.wf.setsampwidth(self.pa.get_sample_size(self.dataformat))
        self.wf.setframerate(self.rate)
                
        def callback(in_data, frame_count, time_info, status):
            self.wf.writeframes(in_data) 
            return (in_data, pyaudio.paContinue)
                
        self.stream = self.pa.open(format=self.dataformat,
                                   channels=self.channels,
                                   rate=self.rate,
                                   input=True,
                                   stream_callback=callback)
        self.stream.start_stream()
        print('Recording started')

    def extract_data(self, s):
        brackets = re.findall(r'\[(.*?)\]', s)
        split_str = re.split(r'\[.*?\]', s)
        word_counts = [len(re.findall(r'\w+', segment)) for segment in split_str if segment]
        result = list(zip(brackets, word_counts))
        cleaned_string = ''.join(split_str).strip()
        return result, cleaned_string

    def stop(self):
        print('Recording finished')
        self.stream.stop_stream()
        self.stream.close()
        self.wf.close()

        file = open("mic.wav", "rb")
        transcription = openai.Audio.transcribe("whisper-1", file)

        transcription = transcription["text"]
        print("Transcription:", transcription)
              
        # Detect negative sentiment and calculate confidence
        is_negative, sentiment_confidence = self.detect_negative_emotion(transcription)
        if is_negative:
            print(f"Negative emotion detected with {sentiment_confidence:.2f}% confidence!")
        else:
            print(f"No negative emotion detected with {sentiment_confidence:.2f}% confidence.")
        
        # Detect keywords and calculate confidence
        keyword_confidence = self.keyword_detection(transcription)
        print(f"Keyword detection confidence: {keyword_confidence:.2f}%")
    
    def detect_negative_emotion(self, transcription):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an emotion analysis assistant."},
                {"role": "user", "content": f"Analyze the following text for negative emotions: {transcription}"}
            ],
            max_tokens=50
        )
        sentiment = response['choices'][0]['message']['content'].strip().lower()

        # Assuming GPT mentions the confidence in sentiment, extract it
        # If GPT doesn't mention it explicitly, return a general confidence level
        if "negative" in sentiment:
            confidence = 100  # Assume a standard confidence (or extract if possible)
            return True, confidence
        else:
            confidence = 0  # Assume a standard confidence for neutral/positive (adjust as needed)
            return False, confidence

    def keyword_detection(self, transcription):
        emergency_keywords = ['emergency', 'help', 'accident', 'fire', 'danger']
        found_keywords = [word for word in emergency_keywords if word in transcription.lower()]

        # Calculate keyword detection confidence
        total_keywords = len(emergency_keywords)
        found_count = len(found_keywords)
        if total_keywords > 0:
            confidence = (found_count / total_keywords) * 100
        else:
            confidence = 0.0
        
        if found_keywords:
            print(f"Emergency detected! Keywords found: {', '.join(found_keywords)}")
        else:
            print("No emergency detected.")
        
        return confidence

def listen_and_process():
    recorder = Recorder("mic.wav")
    
    while True:
        recorder.start()
        time.sleep(3)  # Listen for 3 seconds
        recorder.stop()

if __name__ == '__main__':
    listen_and_process()

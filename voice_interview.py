import streamlit as st
from streamlit_mic_recorder import mic_recorder
import os
import io
from gtts import gTTS
import pygame
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
import uuid

# Load Gemini API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# UI
st.set_page_config(page_title="🎙 Voice Chatbot", layout="centered")
st.title("🎙 Voice to Voice Chatbot using Gemini")

# Save audio as .wav
def save_audio_bytes_as_wav(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        return f.name

# Transcribe voice to text
def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        return "Error connecting to recognition service."

# Get response from Gemini
def chat_with_gemini(prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# Speak bot reply
def speak(text):
    tts = gTTS(text)
    filename = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.quit()
    os.remove(filename)

# Mic recorder
audio = mic_recorder(start_prompt="🎙️ Speak now", stop_prompt="🛑 Stop", just_once=True, use_container_width=True)

if audio:
    st.info("📝 Transcribing...")
    audio_path = save_audio_bytes_as_wav(audio["bytes"])
    user_text = speech_to_text(audio_path)
    st.success(f"🗣 You said: {user_text}")

    st.info("🤖 Thinking...")
    reply = chat_with_gemini(user_text)
    st.success(f"🤖 Bot: {reply}")

    speak(reply)

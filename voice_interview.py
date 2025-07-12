import streamlit as st
import PyPDF2
from docx import Document
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
import asyncio
import json
import os
import pandas as pd
import requests
from dotenv import load_dotenv
import time
import nest_asyncio
import datetime
from gtts import gTTS
import io
import base64
from pydub import AudioSegment
import speech_recognition as sr
import logging
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from typing import List, Union

# Fix for experimental_rerun (workaround for older streamlit-webrtc versions)
if not hasattr(st, 'experimental_rerun'):
    st.experimental_rerun = st.rerun  # Redirect to st.rerun[](https://github.com/monarch-initiative/curategpt/issues/99)

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Apply nest_asyncio for running async functions in Streamlit Cloud
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Retrieve API key and recruiter password
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY environment variable not set. Please set it in your .env file.")

RECRUITER_PASSWORD = os.getenv("RECRUITER_PASSWORD", "admin123")

# --- Audio Processor for WebRTC ---
class AudioBufferProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_chunks: List[bytes] = []
        self.start_time = time.time()
    
    def recv(self, frame) -> bytes:
        logging.info(f"Received audio frame: {len(frame)} bytes")
        self.audio_chunks.append(frame.tobytes())  # Convert frame to bytes
        return frame
    
    def get_audio_data(self) -> bytes:
        return b"".join(self.audio_chunks)

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribes audio from bytes using Google Speech Recognition."""
    logging.info(f"Received audio bytes length: {len(audio_bytes)}")
    r = sr.Recognizer()
    try:
        audio_file = BytesIO(audio_bytes)
        # Assume 16-bit, 44.1kHz, mono audio (common for WebRTC)
        audio = AudioSegment.from_raw(audio_file, sample_width=2, frame_rate=44100, channels=1)
        logging.info(f"AudioSegment: sample_width={audio.sample_width}, frame_rate={audio.frame_rate}, channels={audio.channels}")
        
        wav_file = BytesIO()
        audio.export(wav_file, format="wav")
        wav_file.seek(0)
        
        with sr.AudioFile(wav_file) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Speech recognition service error: {e}"
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        return f"An unexpected error occurred during transcription: {e}"

def record_audio_webrtc():
    """Streamlit UI for client-side audio recording using webrtc_streamer."""
    for var in ['transcribed_text', 'webrtc_audio_data']:
        if var not in st.session_state:
            st.session_state[var] = "" if var == 'transcribed_text' else None
    
    try:
        webrtc_ctx = webrtc_streamer(
            key="audio_recorder",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioBufferProcessor,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["turn:turn.anyfirewall.com:443?transport=tcp"], "username": "user", "credential": "pass"},
                    {"urls": ["turn:turn.bistri.com:80"], "username": "home", "credential": "home"}
                ],
                "iceTransportPolicy": "all"
            },
            media_stream_constraints={"video": False, "audio": True},
            async_processing=True
        )
        if not webrtc_ctx.state.playing:
            st.warning("WebRTC stream not active. Please ensure microphone access is granted, check your network, or use the text input below.")
            return ""
    except Exception as e:
        st.error(f"WebRTC initialization failed: {str(e)}. Falling back to text input.")
        return ""
    
    if webrtc_ctx.audio_processor:
        if st.button("📝 Transcribe Recorded Audio"):
            with st.spinner("Transcribing audio..."):
                audio_data_bytes = webrtc_ctx.audio_processor.get_audio_data()
                if audio_data_bytes:
                    st.session_state.transcribed_text = transcribe_audio_bytes(audio_data_bytes)
                    st.session_state.webrtc_audio_data = audio_data_bytes
                    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
                        st.toast("Transcription complete!")
                    else:
                        st.error(st.session_state.transcribed_text)
                else:
                    st.warning("No audio recorded yet. Please start recording and speak.")
    
    # Fallback text input if WebRTC fails
    text_answer = st.text_area("Or type your answer here if recording fails:", height=150, key=f"text_ans_{st.session_state.current_question_index}")
    
    # Use transcribed text if available, otherwise use typed text
    final_answer = st.session_state.transcribed_text if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not",)) else text_answer.strip()
    
    if final_answer and st.button("Submit Answer (Text or Recorded)", key=f"submit_{st.session_state.current_question_index}"):
        st.session_state.interview_data['qa'].append({
            "question": st.session_state.dynamic_questions[st.session_state.current_question_index],
            "answer": final_answer,
            "audio_file_bytes": st.session_state.webrtc_audio_data
        })
        st.session_state.current_question_index += 1
        st.session_state.audio_question_played = False
        st.session_state.transcribed_text = ""
        st.session_state.webrtc_audio_data = None
        st.rerun()
    
    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
        st.text_area("Transcribed Text", 
                     value=st.session_state.transcribed_text, 
                     height=150,
                     key=f"transcribed_{st.session_state.current_question_index}")
        
        if st.session_state.webrtc_audio_data:
            st.audio(st.session_state.webrtc_audio_data, format='audio/wav', start_time=0)
    
    return final_answer
    
    return final_answer
    
    if webrtc_ctx.audio_processor:
        if st.button("📝 Transcribe Recorded Audio"):
            with st.spinner("Transcribing audio..."):
                audio_data_bytes = webrtc_ctx.audio_processor.get_audio_data()
                if audio_data_bytes:
                    st.session_state.transcribed_text = transcribe_audio_bytes(audio_data_bytes)
                    st.session_state.webrtc_audio_data = audio_data_bytes
                    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
                        st.toast("Transcription complete!")
                    else:
                        st.error(st.session_state.transcribed_text)
                else:
                    st.warning("No audio recorded yet. Please start recording and speak.")
    
    # Fallback text input if WebRTC fails
    text_answer = st.text_area("Or type your answer here if recording fails:", height=150, key=f"text_ans_{st.session_state.current_question_index}")
    
    # Use transcribed text if available, otherwise use typed text
    final_answer = st.session_state.transcribed_text if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not",)) else text_answer.strip()
    
    if final_answer and st.button("Submit Answer (Text or Recorded)", key=f"submit_{st.session_state.current_question_index}"):
        st.session_state.interview_data['qa'].append({
            "question": st.session_state.dynamic_questions[st.session_state.current_question_index],
            "answer": final_answer,
            "audio_file_bytes": st.session_state.webrtc_audio_data
        })
        st.session_state.current_question_index += 1
        st.session_state.audio_question_played = False
        st.session_state.transcribed_text = ""
        st.session_state.webrtc_audio_data = None
        st.rerun()
    
    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
        st.text_area("Transcribed Text", 
                     value=st.session_state.transcribed_text, 
                     height=150,
                     key=f"transcribed_{st.session_state.current_question_index}")
        
        if st.session_state.webrtc_audio_data:
            st.audio(st.session_state.webrtc_audio_data, format='audio/wav', start_time=0)
    
    return final_answer
    
    if webrtc_ctx.audio_processor:
        if st.button("📝 Transcribe Recorded Audio"):
            with st.spinner("Transcribing audio..."):
                audio_data_bytes = webrtc_ctx.audio_processor.get_audio_data()
                if audio_data_bytes:
                    st.session_state.transcribed_text = transcribe_audio_bytes(audio_data_bytes)
                    st.session_state.webrtc_audio_data = audio_data_bytes
                    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
                        st.toast("Transcription complete!")
                    else:
                        st.error(st.session_state.transcribed_text)
                else:
                    st.warning("No audio recorded yet. Please start recording and speak.")
    
    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
        st.text_area("Transcribed Text", 
                     value=st.session_state.transcribed_text, 
                     height=150,
                     key=f"transcribed_{st.session_state.current_question_index}")
        
        if st.session_state.webrtc_audio_data:
            st.audio(st.session_state.webrtc_audio_data, format='audio/wav', start_time=0)
    
    return st.session_state.transcribed_text
    
    if webrtc_ctx.audio_processor:
        if st.button("📝 Transcribe Recorded Audio"):
            with st.spinner("Transcribing audio..."):
                audio_data_bytes = webrtc_ctx.audio_processor.get_audio_data()
                if audio_data_bytes:
                    st.session_state.transcribed_text = transcribe_audio_bytes(audio_data_bytes)
                    st.session_state.webrtc_audio_data = audio_data_bytes
                    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
                        st.toast("Transcription complete!")
                    else:
                        st.error(st.session_state.transcribed_text)
                else:
                    st.warning("No audio recorded yet. Please start recording and speak.")
    
    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
        st.text_area("Transcribed Text", 
                     value=st.session_state.transcribed_text, 
                     height=150,
                     key=f"transcribed_{st.session_state.current_question_index}")
        
        if st.session_state.webrtc_audio_data:
            st.audio(st.session_state.webrtc_audio_data, format='audio/wav', start_time=0)
    
    return st.session_state.transcribed_text
    
    if webrtc_ctx.audio_processor:
        if st.button("📝 Transcribe Recorded Audio"):
            with st.spinner("Transcribing audio..."):
                audio_data_bytes = webrtc_ctx.audio_processor.get_audio_data()
                if audio_data_bytes:
                    st.session_state.transcribed_text = transcribe_audio_bytes(audio_data_bytes)
                    st.session_state.webrtc_audio_data = audio_data_bytes
                    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
                        st.toast("Transcription complete!")
                    else:
                        st.error(st.session_state.transcribed_text)
                else:
                    st.warning("No audio recorded yet. Please start recording and speak.")
    
    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
        st.text_area("Transcribed Text", 
                     value=st.session_state.transcribed_text, 
                     height=150,
                     key=f"transcribed_{st.session_state.current_question_index}")
        
        if st.session_state.webrtc_audio_data:
            st.audio(st.session_state.webrtc_audio_data, format='audio/wav', start_time=0)
    
    return st.session_state.transcribed_text
    
    if webrtc_ctx.audio_processor:
        if st.button("📝 Transcribe Recorded Audio"):
            with st.spinner("Transcribing audio..."):
                audio_data_bytes = webrtc_ctx.audio_processor.get_audio_data()
                if audio_data_bytes:
                    st.session_state.transcribed_text = transcribe_audio_bytes(audio_data_bytes)
                    st.session_state.webrtc_audio_data = audio_data_bytes
                    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
                        st.toast("Transcription complete!")
                    else:
                        st.error(st.session_state.transcribed_text)
                else:
                    st.warning("No audio recorded yet. Please start recording and speak.")
    
    if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
        st.text_area("Transcribed Text", 
                     value=st.session_state.transcribed_text, 
                     height=150,
                     key=f"transcribed_{st.session_state.current_question_index}")
        
        if st.session_state.webrtc_audio_data:
            st.audio(st.session_state.webrtc_audio_data, format='audio/wav', start_time=0)
    
    return st.session_state.transcribed_text

def text_to_speech(text, lang='en'):
    """Converts text to speech using gTTS and returns audio bytes."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}. Ensure internet connectivity for gTTS.")
        return None

def autoplay_audio(audio_bytes):
    """Embeds an audio player that autoplays the provided audio bytes."""
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        audio_html = f"""
        <audio controls autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
        </audio>
        """
        st.components.v1.html(audio_html, height=50)

def extract_text_from_document(uploaded_file):
    """Extracts text from PDF, DOCX, or TXT files."""
    file_name = uploaded_file.name.lower()
    text = ""
    try:
        if file_name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif file_name.endswith((".doc", ".docx")):
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file_name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
        else:
            st.warning("Unsupported file type. Please upload PDF, DOCX, or TXT.")
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {str(e)}")
        return ""
    return text

def load_shortlisted_candidates_from_excel(uploaded_file):
    """Loads candidate data from an Excel file."""
    try:
        df = pd.read_excel(uploaded_file)
        if 'Name' not in df.columns:
            st.error("Excel file must contain a column named 'Name' for candidate names.")
            return None
        if 'Job Description' not in df.columns:
            st.warning("Excel file does not contain a 'Job Description' column. Questions will be more general.")
            df['Job Description'] = ""
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None

def format_transcript_for_download(interview_data):
    """Formats the interview data into a human-readable transcript string."""
    candidate_name = interview_data.get("candidate_name", "N/A Candidate")
    timestamp = interview_data.get("timestamp", "N/A Date")
    total_score = interview_data.get("total_score", "N/A")
    jd = interview_data.get("jd", "Not provided.")
    resume_text = interview_data.get("verification_text", "Not provided.")

    transcript_lines = [
        f"--- AI Interview Transcript for {candidate_name} ---",
        f"Interview Date: {timestamp}",
        f"Overall Score: {total_score}/30",
        f"--------------------------------------------------",
        f"\nJob Description:",
        f"-------------------",
        jd,
        f"-------------------",
        f"\nCandidate's Resume/Verification Text (Excerpt):",
        f"-------------------",
        f"{resume_text[:1000]}...",
        f"-------------------",
        f"\nDetailed Questions & Answers:"
    ]
    for i, qa in enumerate(interview_data.get("qa", []), 1):
        question = qa.get("question", "N/A Question")
        answer = qa.get("answer", "N/A Answer")
        score = qa.get("score", "N/A")
        feedback = qa.get("feedback", "No specific feedback provided.")
        transcript_lines.extend([
            f"\nQuestion {i}: {question}",
            f"Candidate Answer: {answer}",
            f"Score: {score}/10",
            f"Feedback: {feedback}",
            f"---"
        ])
    transcript_lines.append(f"\n--- End of Interview Transcript ---")
    return "\n".join(transcript_lines)

async def generate_interview_questions(jd):
    """Generates interview questions using the Gemini API."""
    try:
        provider = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY
        )
        model = OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
        set_tracing_disabled(disabled=True)

        agent = Agent(
            name="Question Generator",
            instructions=f"""
            Generate 3 concise interview questions based on the following Job Description.
            Ensure the questions are relevant to the job role and can be answered by a candidate.
            If the Job Description is empty or very general, generate general but professional interview questions.
            
            Job Description: "{jd}"
            
            Output format:
            {{
                "questions": [
                    "Question 1",
                    "Question 2", 
                    "Question 3"
                ]
            }}
            """,
            model=model,
        )

        result = Runner.run_streamed(starting_agent=agent, input="Generate questions")
        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta

        json_string = full_response.strip().strip('```json').strip('```')
        response_json = json.loads(json_string)
        return response_json["questions"]
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from question generation: {e}. Raw response: {full_response}")
        return ["Tell me about your experience.", "What are your strengths?", "Why are you interested in this role?"]
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return ["Tell me about yourself.", "Describe your experience.", "Why are you interested in this role?"]

async def conduct_interview(questions_answers, jd, resume_text):
    """Evaluates candidate's answers using the Gemini API."""
    try:
        provider = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY
        )
        model = OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
        set_tracing_disabled(disabled=True)

        interview_input_str = "\n\n".join([f"Question: {qa['question']}\nAnswer: {qa['answer']}" for qa in questions_answers])

        agent = Agent(
            name="AI Interview Evaluator",
            instructions=f"""
            You are an AI Interviewer. Evaluate the candidate's responses based on the provided Job Description and their Resume content.
            For each question, provide a score (1-10) and constructive feedback.
            The total score should be the sum of individual question scores.

            Job Description: "{jd}"
            Resume Content: "{resume_text}"

            Here are the questions and candidate's answers:
            {interview_input_str}

            Output format:
            {{
                "questions": [
                    {{
                        "question": "The exact question asked",
                        "answer": "The exact answer given by candidate",
                        "score": 0-10,
                        "feedback": "Constructive feedback for this answer."
                    }},
                    {{
                        "question": "...",
                        "answer": "...",
                        "score": 0-10,
                        "feedback": "..."
                    }}
                ],
                "total_score": 0-30
            }}
            """,
            model=model,
        )

        st.info("Analyzing responses and generating feedback... This might take a moment.")
        result = Runner.run_streamed(starting_agent=agent, input="Evaluate the interview.")
        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta

        json_string = full_response.strip().strip('```json').strip('```')
        response_json = json.loads(json_string)
        
        for i, qa in enumerate(st.session_state.interview_data["qa"]):
            if i < len(response_json["questions"]):
                ai_qa = response_json["questions"][i]
                qa["score"] = int(ai_qa.get("score", 0))
                qa["feedback"] = ai_qa.get("feedback", "No feedback provided.")
        
        st.session_state.interview_data["total_score"] = int(response_json.get("total_score", 0))
        
        st.session_state.interviews[st.session_state.interview_data["candidate_name"]] = {
            "timestamp": st.session_state.interview_data["timestamp"],
            "total_score": st.session_state.interview_data["total_score"],
            "qa": st.session_state.interview_data["qa"],
            "jd": st.session_state.interview_data["jd"],
            "resume_text": st.session_state.interview_data["verification_text"]
        }
        st.session_state.interview_processed_successfully = True
        st.success("Interview evaluation complete!")
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from evaluation: {e}. Raw response: {full_response}")
        st.session_state.interview_started_processing = False
        st.session_state.interview_processed_successfully = False
    except Exception as e:
        st.error(f"Error during interview evaluation: {str(e)}")
        st.session_state.interview_started_processing = False
        st.session_state.interview_processed_successfully = False

def recruiter_login_logic():
    """Handles the recruiter login process."""
    st.subheader("🔑 Recruiter Login")
    password = st.text_input("Password", type="password", key="recruiter_password")
    if st.button("Login", key="recruiter_login_btn"):
        if password == RECRUITER_PASSWORD:
            st.session_state.current_page = "recruiter_dashboard"
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    if st.button("Back to Candidate Verification", key="back_to_verification_from_login"):
        st.session_state.current_page = "verification"
        st.rerun()

# Initialize session state variables
for key, default_value in {
    'current_page': "verification",
    'interview_data': {},
    'shortlisted_df': None,
    'interviews': {},
    'current_question_index': 0,
    'interview_started_processing': False,
    'interview_processed_successfully': False,
    'authenticated': False,
    'dynamic_questions': [],
    'audio_question_played': False,
    'transcribed_text': "",
    'webrtc_audio_data': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Main Streamlit Application Layout ---
st.title("AI Interview Portal 🚀")
st.markdown("---")

# Page: Candidate Verification
if st.session_state.current_page == "verification":
    st.header("📝 Candidate Verification")
    with st.expander("Candidate Information", expanded=True):
        full_name = st.text_input("Full Name", value=st.session_state.interview_data.get("candidate_name", ""), help="Enter your full name as it appears in the shortlist.")
        uploaded_file = st.file_uploader("Upload Resume/CV", type=["pdf", "doc", "docx", "txt"], help="Upload your resume in PDF, DOCX, or TXT format.")

    st.markdown("---")
    st.subheader("Recruiter Actions")
    uploaded_excel = st.file_uploader("Upload Shortlisted Candidates (Excel)", type=["xlsx"], help="Recruiters: Upload an Excel file containing candidate names and job descriptions.")
    
    if uploaded_excel and st.button("Load Shortlisted List"):
        st.session_state.shortlisted_df = load_shortlisted_candidates_from_excel(uploaded_excel)
        if st.session_state.shortlisted_df is not None:
            st.success("Candidates loaded successfully!")
            st.dataframe(st.session_state.shortlisted_df.head(), use_container_width=True)

    st.markdown("---")
    col_start, col_recruiter = st.columns([3, 1])
    with col_start:
        if st.button("Start Interview", type="primary"):
            if not full_name.strip():
                st.error("Please provide your name to start the interview.")
            elif st.session_state.shortlisted_df is None:
                st.error("Shortlist not loaded. Please upload an Excel file with candidate names.")
            else:
                candidate_row = st.session_state.shortlisted_df[st.session_state.shortlisted_df['Name'].str.strip().str.lower() == full_name.strip().lower()]
                
                if candidate_row.empty:
                    st.error("Your name was not found in the shortlist. Please check the name or contact the recruiter.")
                else:
                    if uploaded_file:
                        verification_text = extract_text_from_document(uploaded_file)
                        if verification_text.strip():
                            candidate_jd = candidate_row['Job Description'].iloc[0] if 'Job Description' in candidate_row.columns and not candidate_row['Job Description'].empty else ""
                            
                            with st.spinner("Generating personalized interview questions..."):
                                st.session_state.dynamic_questions = asyncio.run(generate_interview_questions(candidate_jd))
                            
                            if st.session_state.dynamic_questions:
                                st.session_state.interview_data = {
                                    "candidate_name": full_name.strip(),
                                    "jd": candidate_jd.strip(),
                                    "verification_text": verification_text.strip(),
                                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "qa": []
                                }
                                st.session_state.current_page = "interview"
                                st.session_state.current_question_index = 0
                                st.session_state.audio_question_played = False
                                st.session_state.transcribed_text = ""
                                st.session_state.webrtc_audio_data = None
                                st.rerun()
                            else:
                                st.error("Failed to generate interview questions. Please try again or check API key.")
                        else:
                            st.error("Could not read content from your resume. Please ensure it's a valid PDF, DOCX, or TXT file and not empty.")
                    else:
                        st.error("Please upload your Resume/CV to proceed with the interview.")
    with col_recruiter:
        if st.button("Recruiter Dashboard"):
            st.session_state.current_page = "recruiter_login"
            st.rerun()

# Page: Interview in Progress
elif st.session_state.current_page == "interview":
    data = st.session_state.interview_data
    st.header(f"Interview: {data['candidate_name']}")
    st.markdown("---")
    
    if st.session_state.interview_processed_successfully:
        st.subheader("✅ Interview Results")
        st.info(f"Overall Score: {data.get('total_score', 'N/A')}/30")
        
        for i, qa in enumerate(data['qa'], 1):
            with st.expander(f"Question {i}: {qa['question']}"):
                st.write(f"**Your Answer:** {qa['answer']}")
                if qa.get('audio_file_bytes'):
                    st.audio(qa['audio_file_bytes'], format='audio/wav')
                st.write(f"**AI Score:** {qa.get('score', 'N/A')}/10")
                st.write(f"**AI Feedback:** {qa.get('feedback', 'No feedback provided.')}")
        
        transcript_content = format_transcript_for_download(data)
        st.download_button(
            label="Download Interview Transcript",
            data=transcript_content,
            file_name=f"{data['candidate_name']}_interview_transcript.txt",
            mime="text/plain",
            help="Download a text file containing all questions, answers, scores, and feedback."
        )

        if st.button("Finish Interview"):
            st.session_state.current_page = "verification"
            st.session_state.current_question_index = 0
            st.session_state.interview_started_processing = False
            st.session_state.interview_processed_successfully = False
            st.session_state.interview_data = {}
            st.session_state.dynamic_questions = []
            st.session_state.audio_question_played = False
            st.session_state.transcribed_text = ""
            st.session_state.webrtc_audio_data = None
            st.rerun()
            
    else:
        if st.session_state.current_question_index < len(st.session_state.dynamic_questions):
            question = st.session_state.dynamic_questions[st.session_state.current_question_index]
            st.subheader(f"Question {st.session_state.current_question_index + 1}/{len(st.session_state.dynamic_questions)}")
            
            if not st.session_state.audio_question_played:
                audio = text_to_speech(question)
                if audio:
                    st.write(f"**{question}**")
                    autoplay_audio(audio)
                st.session_state.audio_question_played = True
            else:
                st.write(f"**{question}**")
            
            st.write("Record your answer below:")
            transcription = record_audio_webrtc()
            
            text_answer = st.text_area("Or type your answer here:", 
                                       value=st.session_state.transcribed_text if st.session_state.transcribed_text and not st.session_state.transcribed_text.startswith(("Could not", "Speech recognition service error", "An unexpected error")) else "",
                                       height=150,
                                       key=f"text_ans_{st.session_state.current_question_index}")
            
            final_answer = ""
            if transcription and not transcription.startswith(("Could not", "Speech recognition service error", "An unexpected error")):
                final_answer = transcription
            elif text_answer.strip():
                final_answer = text_answer
            
            if st.button("Submit Answer", type="primary"):
                if not final_answer.strip():
                    st.error("Please provide a valid answer either by recording or typing.")
                else:
                    st.session_state.interview_data['qa'].append({
                        "question": question,
                        "answer": final_answer.strip(),
                        "audio_file_bytes": st.session_state.webrtc_audio_data
                    })
                    
                    st.session_state.current_question_index += 1
                    st.session_state.audio_question_played = False
                    st.session_state.transcribed_text = ""
                    st.session_state.webrtc_audio_data = None
                    
                    if st.session_state.current_question_index >= len(st.session_state.dynamic_questions):
                        st.session_state.interview_started_processing = True
                        with st.spinner("Evaluating your interview... This may take a moment."):
                            asyncio.run(conduct_interview(
                                st.session_state.interview_data['qa'],
                                st.session_state.interview_data['jd'],
                                st.session_state.interview_data['verification_text']
                            ))
                        st.rerun()
                    else:
                        st.rerun()
        
        elif st.session_state.interview_started_processing and not st.session_state.interview_processed_successfully:
            st.info("Interview evaluation is in progress. Please wait...")

# Page: Recruiter Login
elif st.session_state.current_page == "recruiter_login":
    recruiter_login_logic()

# Page: Recruiter Dashboard
elif st.session_state.current_page == "recruiter_dashboard":
    if not st.session_state.authenticated:
        st.warning("Please log in to access the Recruiter Dashboard.")
        recruiter_login_logic()
    else:
        st.header("📊 Recruiter Dashboard")
        
        if st.session_state.interviews:
            st.subheader("Completed Interviews")
            
            interview_list = []
            for name, data in st.session_state.interviews.items():
                interview_list.append({
                    "Candidate Name": name,
                    "Date": data.get("timestamp", "N/A"),
                    "Total Score": data.get("total_score", "N/A"),
                    "Job Description": data.get("jd", "N/A"),
                    "Questions Answered": len(data.get("qa", []))
                })
            
            interviews_df = pd.DataFrame(interview_list)
            st.dataframe(interviews_df, use_container_width=True)

            selected_candidate = st.selectbox(
                "Select a candidate to view detailed interview:",
                options=[""] + list(st.session_state.interviews.keys()),
                key="dashboard_candidate_select"
            )

            if selected_candidate:
                data = st.session_state.interviews[selected_candidate]
                st.subheader(f"Detailed Interview for {selected_candidate}")
                st.info(f"Overall Score: {data.get('total_score', 'N/A')}/30")
                st.write(f"**Job Description:** {data.get('jd', 'Not provided.')}")
                st.write(f"**Resume Excerpt:** {data.get('resume_text', 'Not provided.')[:500]}...")
                
                for i, qa in enumerate(data['qa'], 1):
                    with st.expander(f"Question {i}: {qa['question']}"):
                        st.write(f"**Candidate Answer:** {qa['answer']}")
                        if qa.get('audio_file_bytes'):
                            st.audio(qa['audio_file_bytes'], format='audio/wav')
                        st.write(f"**AI Score:** {qa.get('score', 'N/A')}/10")
                        st.write(f"**AI Feedback:** {qa.get('feedback', 'No feedback provided.')}")
                
                transcript_content = format_transcript_for_download(data)
                st.download_button(
                    label=f"Download {selected_candidate} Transcript",
                    data=transcript_content,
                    file_name=f"{selected_candidate}_interview_transcript.txt",
                    mime="text/plain"
                )

        else:
            st.info("No interviews have been completed yet. Conduct interviews to see results here.")
            
        st.markdown("---")
        if st.button("Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.current_page = "verification"
            st.rerun()

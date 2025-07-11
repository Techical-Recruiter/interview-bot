import streamlit as st
import PyPDF2
from docx import Document
import asyncio
import json
import os
import pandas as pd
import time
import datetime
from gtts import gTTS
import io
import base64
import wave
import pyaudio
import speech_recognition as sr
from io import BytesIO
from dotenv import load_dotenv
import requests
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORDING_FILE = "temp_recording.wav"

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "verification"
if 'interview_data' not in st.session_state:
    st.session_state.interview_data = {}
if 'shortlisted_df' not in st.session_state:
    st.session_state.shortlisted_df = None
if 'interviews' not in st.session_state:
    st.session_state.interviews = {}
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'interview_started_processing' not in st.session_state:
    st.session_state.interview_started_processing = False
if 'interview_processed_successfully' not in st.session_state:
    st.session_state.interview_processed_successfully = False
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'dynamic_questions' not in st.session_state:
    st.session_state.dynamic_questions = []
if 'audio_question_played' not in st.session_state:
    st.session_state.audio_question_played = False
if 'recorder' not in st.session_state:
    st.session_state.recorder = None
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""

class AudioRecorder:
    def __init__(self):
        try:
            self.audio = pyaudio.PyAudio()
            self.frames = []
            self.is_recording = False
            self.stream = None
            self.initialized = True
        except Exception as e:
            st.error(f"Audio initialization failed: {str(e)}")
            self.initialized = False

    def start_recording(self):
        if not self.initialized:
            return False
        try:
            self.frames = []
            self.is_recording = True
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self.callback
            )
            self.stream.start_stream()
            return True
        except Exception as e:
            st.error(f"Recording failed: {str(e)}")
            return False

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.save_recording()

    def save_recording(self):
        try:
            wf = wave.open(RECORDING_FILE, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return True
        except Exception as e:
            st.error(f"Failed to save recording: {str(e)}")
            return False

    def transcribe_recording(self):
        try:
            r = sr.Recognizer()
            with sr.AudioFile(RECORDING_FILE) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Speech service error"
        except Exception as e:
            return f"Transcription error: {str(e)}"

def record_audio():
    if st.session_state.recorder is None:
        st.session_state.recorder = AudioRecorder()
    
    if not st.session_state.recorder.initialized:
        st.error("Microphone not available")
        return ""
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 Start Recording", disabled=st.session_state.recording):
            if st.session_state.recorder.start_recording():
                st.session_state.recording = True
                st.session_state.audio_file = None
                st.session_state.transcribed_text = ""
                st.toast("Recording started...")
    
    with col2:
        if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording):
            st.session_state.recorder.stop_recording()
            st.session_state.recording = False
            st.session_state.audio_file = RECORDING_FILE
            st.toast("Recording stopped")
    
    if st.session_state.audio_file:
        st.audio(st.session_state.audio_file, format='audio/wav')
        
        if st.button("📝 Transcribe"):
            with st.spinner("Transcribing..."):
                st.session_state.transcribed_text = st.session_state.recorder.transcribe_recording()
                if st.session_state.transcribed_text.startswith(("Could not", "Speech service", "Transcription")):
                    st.error(st.session_state.transcribed_text)
                else:
                    st.toast("Transcription complete")
    
    if st.session_state.transcribed_text:
        return st.text_area("Transcribed Answer", 
                          value=st.session_state.transcribed_text,
                          height=150,
                          key=f"transcribed_{st.session_state.current_question_index}")
    return ""

def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"TTS error: {str(e)}")
        return None

def autoplay_audio(audio_bytes):
    audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
    audio_html = f"""
    <audio controls autoplay>
    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """
    st.components.v1.html(audio_html, height=50)

def extract_text_from_document(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            return "".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif uploaded_file.name.lower().endswith((".doc", ".docx")):
            doc = Document(uploaded_file)
            return "\n".join(para.text for para in doc.paragraphs)
        elif uploaded_file.name.lower().endswith(".txt"):
            return uploaded_file.read().decode("utf-8")
        else:
            st.warning("Unsupported file type")
            return ""
    except Exception as e:
        st.error(f"File error: {str(e)}")
        return ""

def load_shortlisted_candidates(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        if 'Name' not in df.columns:
            st.error("Missing 'Name' column")
            return None
        if 'Job Description' not in df.columns:
            df['Job Description'] = ""
        return df
    except Exception as e:
        st.error(f"Excel error: {str(e)}")
        return None

async def generate_questions(jd):
    try:
        provider = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY
        )
        model = OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
        
        agent = Agent(
            name="Question Generator",
            instructions=f"""
            Generate 3 interview questions based on this Job Description:
            "{jd}"
            
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

        response_json = json.loads(full_response.strip().strip('```json').strip('```'))
        return response_json["questions"]
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return [
            "Tell me about yourself.",
            "Describe your experience.",
            "Why are you interested in this role?"
        ]

async def evaluate_answers(answers, jd, resume):
    try:
        provider = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY
        )
        model = OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
        
        agent = Agent(
            name="AI Interview Evaluator",
            instructions=f"""
            Evaluate interview responses based on:
            - Resume: "{resume}"
            - Job Description: "{jd}"
            
            For each question, provide:
            - score (1-10)
            - constructive feedback
            
            Output format:
            {{
                "questions": [
                    {{
                        "question": "...",
                        "answer | "..." | 
                        "score": 0-10,
                        "feedback": "..."
                    }}
                ],
                "total_score": 0-30
            }}
            """,
            model=model,
        )

        interview_input = "\n\n".join([f"Question {i+1}: {qa['question']}\nAnswer: {qa['answer']}" for i, qa in enumerate(answers)])
        
        result = Runner.run_streamed(starting_agent=agent, input=interview_input)
        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta

        response_json = json.loads(full_response.strip().strip('```json').strip('```'))
        
        for i, qa in enumerate(answers):
            if i < len(response_json["questions"]):
                ai_qa = response_json["questions"][i]
                qa["score"] = int(ai_qa.get("score", 0))
                qa["feedback"] = ai_qa.get("feedback", "No feedback")
        
        return int(response_json.get("total_score", 0))
    except Exception as e:
        st.error(f"Error in evaluation: {str(e)}")
        # Fallback to simple evaluation
        scores = []
        for qa in answers:
            score = min(10, max(1, len(qa['answer']) // 20))  # Simple length-based scoring
            feedback = "Good answer" if score > 5 else "Could be more detailed"
            qa['score'] = score
            qa['feedback'] = feedback
            scores.append(score)
        return sum(scores)

def format_transcript_for_download(interview_data):
    candidate_name = interview_data.get("candidate_name", "N/A Candidate")
    timestamp = interview transcript_data.get("timestamp", "N/A Date")
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

# Main App Interface
st.title("AI Interview Portal")
st.markdown("---")

if st.session_state.current_page == "verification":
    st.header("Candidate Verification")
    
    with st.expander("Your Information"):
        name = st.text_input("Full Name")
        resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    
    st.markdown("---")
    st.subheader("Recruiter Section")
    shortlist = st.file_uploader("Upload Shortlist", type=["xlsx"])
    
    if shortlist and st.button("Load Candidates"):
        st.session_state.shortlisted_df = load_shortlisted_candidates(shortlist)
        if st.session_state.shortlisted_df is not None:
            st.success(f"Loaded {len(st.session_state.shortlisted_df)} candidates")
    
    if st.button("Start Interview"):
        if not name.strip():
            st.error("Please enter your name")
        elif st.session_state.shortlisted_df is None:
            st.error("Shortlist not loaded")
        else:
            candidate = st.session_state.shortlisted_df[
                st.session_state.shortlisted_df['Name'].str.strip().str.lower() == name.strip().lower()
            ]
            if candidate.empty:
                st.error("Name not found in shortlist")
            elif not resume:
                st.error("Please upload resume")
            else:
                resume_text = extract_text_from_document(resume)
                if not resume_text.strip():
                    st.error("Could not read resume")
                else:
                    jd = candidate['Job Description'].iloc[0] if 'Job Description' in candidate.columns else ""
                    st.session_state.dynamic_questions = asyncio.run(generate_questions(jd))
                    st.session_state.interview_data = {
                        "candidate_name": name.strip(),
                        "jd": jd.strip(),
                        "verification_text": resume_text.strip(),
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "qa": []
                    }
                    st.session_state.current_page = "interview"
                    st.rerun()

elif st.session_state.current_page == "interview":
    data = st.session_state.interview_data
    st.header(f"Interview: {data['candidate_name']}")
    st.markdown("---")
    
    if st.session_state.interview_processed_successfully:
        st.subheader("Results")
        st.write(f"Total Score: {data['total_score']}/30")
        
        for i, qa in enumerate(data['qa'], 1):
            with st.expander(f"Question {i}"):
                st.write(f"**Q:** {qa['question']}")
                st.write(f"**A:** {qa['answer']}")
                if qa.get('audio_file'):
                    st.audio(qa['audio_file'])
                st.write(f"**Score:** {qa.get('score', 'N/A')}/10")
                st.write(f"**Feedback:** {qa.get('feedback', 'None')}")
        
        if st.button("Finish"):
            st.session_state.current_page = "verification"
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
            
            st.write("Record your answer:")
            transcription = record_audio()
            
            text_answer = st.text_area("Or type answer", 
                                     height=150,
                                     key=f"text_ans_{st.session_state.current_question_index}")
            
            final_answer = ""
            if transcription and not transcription.startswith(("Could not", "Speech service", "Transcription")):
                final_answer = transcription
            elif text_answer.strip():
                final_answer = text_answer
            
            if st.button("Submit Answer"):
                if not final_answer:
                    st.error("Please provide a valid answer")
                else:
                    data['qa'].append({
                        "question": question,
                        "answer": final_answer.strip(),
                        "audio_file": st.session_state.audio_file
                    })
                    
                    # Reset for next question
                    st.session_state.current_question_index += 1
                    st.session_state.audio_question_played = False
                    st.session_state.recording = False
                    st.session_state.audio_file = None
                    st.session_state.transcribed_text = ""
                    
                    if st.session_state.current_question_index >= len(st.session_state.dynamic_questions):
                        # All questions answered - evaluate
                        st.info("Processing your interview...")
                        data['total_score'] = asyncio.run(evaluate_answers(data['qa'], data['jd'], data['verification_text']))
                        st.session_state.interview_processed_successfully = True
                        st.session_state.interviews[data['candidate_name']] = data
                    st.rerun()
        
        elif not st.session_state.interview_started_processing:
            st.session_state.interview_started_processing = True
            st.rerun()

elif st.session_state.current_page == "recruiter_dashboard":
    if not st.session_state.authenticated:
        st.subheader("Recruiter Login")
        if st.text_input("Password", type="password") == os.getenv("RECRUITER_PW", "admin123"):
            st.session_state.authenticated = True
            st.rerun()
        elif st.button("Back"):
            st.session_state.current_page = "verification"
            st.rerun()
    else:
        st.header("Recruiter Dashboard")
        
        if st.session_state.interviews:
            st.subheader("Completed Interviews")
            for name, data in st.session_state.interviews.items():
                with st.expander(f"{name} - {data['total_score']}/30"):
                    st.write(f"Date: {data['timestamp']}")
                    for i, qa in enumerate(data['qa'], 1):
                        st.write(f"**Q{i}:** {qa['question']}")
                        st.write(f"**A:** {qa['answer']}")
                        st.write(f"Score: {qa.get('score', 'N/A')}/10")
                        st.write(f"Feedback: {qa.get('feedback', 'None')}")
                    st.download_button(
                        label="Download Transcript",
                        data=format_transcript_for_download(data),
                        file_name=f"{name}_interview.txt"
                    )
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.current_page = "verification"
            st.rerun()

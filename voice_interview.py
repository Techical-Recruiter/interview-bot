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
import tempfile
from streamlit_mic_recorder import mic_recorder

# Configure logging if you want (optional)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize nest_asyncio for asyncio compatibility in Streamlit
nest_asyncio.apply()
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RECRUITER_PASSWORD = os.getenv("RECRUITER_PASSWORD") # Make sure this is also in your Streamlit secrets

# --- Database Connection ---
# Use the name you defined in secrets.toml under [connections.NAME]
conn = st.connection("neon_db", type="sql")

# --------- AUDIO RECORDING & TRANSCRIPTION USING streamlit_mic_recorder ------------

def save_audio_bytes_as_wav(audio_bytes):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio_segment.export(f.name, format="wav")
        return f.name

def record_audio():
    if 'audio_bytes' not in st.session_state:
        st.session_state.audio_bytes = None
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""

    audio = mic_recorder(
        start_prompt="🎙️ Speak now",
        stop_prompt="🛑 Stop",
        just_once=True,
        use_container_width=True
    )
    
    if audio:
        st.session_state.audio_bytes = audio["bytes"]
        st.audio(st.session_state.audio_bytes, format="audio/wav")

        audio_path = save_audio_bytes_as_wav(st.session_state.audio_bytes)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                st.session_state.transcribed_text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                st.session_state.transcribed_text = "Sorry, I couldn't understand that."
            except sr.RequestError:
                st.session_state.transcribed_text = "Speech recognition service unavailable."

    if st.session_state.transcribed_text:
        return st.text_area(
            "Transcribed Text (You can edit if needed):", 
            value=st.session_state.transcribed_text,
            height=150,
            key=f"transcribed_{st.session_state.current_question_index}"
        )
    else:
        return ""

def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        return None
    
def autoplay_audio(audio_bytes):
    # Ensure audio_bytes is at the beginning before reading
    audio_bytes.seek(0) 
    audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
    audio_html = f"""
    <audio controls autoplay>
    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    Your browser does not support the audio element.
    </audio>
    """
    st.components.v1.html(audio_html, height=50)

def extract_text_from_document(uploaded_file):
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
    try:
        df = pd.read_excel(uploaded_file)
        if 'Name' not in df.columns:
            st.error("Excel file must contain a column named 'Name' for candidate names.")
            return None
        if 'Job Description' not in df.columns:
            st.warning("Excel file does not contain a 'Job Description' column.")
            df['Job Description'] = ""
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None

# --- NEW: Functions to interact with the database ---
def save_interview_to_db(interview_data):
    try:
        with conn.session as session:
            # Insert into interviews table
            result = session.execute(
                """
                INSERT INTO interviews (candidate_name, interview_timestamp, total_score, job_description, resume_text)
                VALUES (:candidate_name, :interview_timestamp, :total_score, :job_description, :resume_text)
                RETURNING id;
                """,
                params={
                    "candidate_name": interview_data["candidate_name"],
                    "interview_timestamp": interview_data["timestamp"],
                    "total_score": interview_data["total_score"],
                    "job_description": interview_data["jd"],
                    "resume_text": interview_data["verification_text"]
                }
            )
            interview_id = result.scalar_one() # Get the ID of the newly inserted interview

            # Insert into qa_details table for each Q&A
            for qa_item in interview_data.get("qa", []):
                audio_bytes_to_store = qa_item.get("audio_bytes").getvalue() if qa_item.get("audio_bytes") else None
                session.execute(
                    """
                    INSERT INTO qa_details (interview_id, question, answer, score, feedback, audio_bytes)
                    VALUES (:interview_id, :question, :answer, :score, :feedback, :audio_bytes)
                    """,
                    params={
                        "interview_id": interview_id,
                        "question": qa_item["question"],
                        "answer": qa_item["answer"],
                        "score": qa_item.get("score"),
                        "feedback": qa_item.get("feedback"),
                        "audio_bytes": audio_bytes_to_store # Store as bytes
                    }
                )
            session.commit()
        st.success("Interview data saved successfully to database!")
    except Exception as e:
        st.error(f"Error saving interview data to database: {str(e)}")
        logging.error(f"Database save error: {e}")


def load_interviews_from_db():
    interviews = {}
    try:
        with conn.session as session:
            # Fetch all main interview records
            main_interviews_result = session.execute("SELECT * FROM interviews;").fetchall()
            
            for row in main_interviews_result:
                # Convert row to dict for easier access
                interview_dict = {
                    "id": row[0], # Assuming id is the first column
                    "candidate_name": row[1],
                    "timestamp": row[2].strftime("%Y-%m-%d %H:%M:%S") if row[2] else None, # Format datetime
                    "total_score": row[3],
                    "jd": row[4],
                    "resume_text": row[5]
                }
                
                # Fetch Q&A details for this interview
                qa_details_result = session.execute(
                    f"SELECT question, answer, score, feedback, audio_bytes FROM qa_details WHERE interview_id = {interview_dict['id']};"
                ).fetchall()
                
                qa_list = []
                for qa_row in qa_details_result:
                    qa_dict = {
                        "question": qa_row[0],
                        "answer": qa_row[1],
                        "score": qa_row[2],
                        "feedback": qa_row[3],
                        # Convert bytes back to BytesIO for Streamlit audio playback
                        "audio_bytes": io.BytesIO(qa_row[4]) if qa_row[4] else None
                    }
                    qa_list.append(qa_dict)
                
                interview_dict["qa"] = qa_list
                interviews[interview_dict["candidate_name"]] = interview_dict
    except Exception as e:
        st.error(f"Error loading interview data from database: {str(e)}")
        logging.error(f"Database load error: {e}")
    return interviews

# --- End of NEW database functions ---


def format_transcript_for_download(interview_data):
    candidate_name = interview_data.get("candidate_name", "N/A Candidate")
    timestamp = interview_data.get("timestamp", "N/A Date")
    total_score = interview_data.get("total_score", "N/A")
    jd = interview_data.get("jd", "Not provided.")
    resume_text = interview_data.get("resume_text", "Not provided.") # Changed from verification_text

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
        logging.error(f"Question generation error: {e}")
        return ["Tell me about yourself.", "Describe your experience.", "Why are you interested in this role?"]

async def conduct_interview(questions, resume_text):
    try:
        provider = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY
        )
        model = OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
        set_tracing_disabled(disabled=True)

        agent = Agent(
            name="AI Interview Evaluator",
            instructions=f"""
            Evaluate interview responses based on:
            - Resume: "{resume_text}"
            - Job Description: "{st.session_state.interview_data['jd']}"
            
            For each question, provide:
            - score (1-10)
            - constructive feedback
            
            Output format:
            {{
                "questions": [
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

        qa_data = st.session_state.interview_data["qa"]
        interview_input = "\n\n".join([f"Question {i+1}: {q['question']}\nAnswer: {q['answer']}" for i, q in enumerate(qa_data)])

        st.info("Analyzing responses...")
        result = Runner.run_streamed(starting_agent=agent, input=interview_input)
        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta

        response_json = json.loads(full_response.strip().strip('```json').strip('```'))
        
        for i, qa in enumerate(st.session_state.interview_data["qa"]):
            if i < len(response_json["questions"]):
                ai_qa = response_json["questions"][i]
                qa["score"] = int(ai_qa.get("score", 0))
                qa["feedback"] = ai_qa.get("feedback", "No feedback")
        
        st.session_state.interview_data["total_score"] = int(response_json.get("total_score", 0))
        
        # --- Critical Change: Save to DB here instead of just session_state.interviews ---
        save_interview_to_db(st.session_state.interview_data)
        # --- End Critical Change ---

        st.session_state.interview_processed_successfully = True
        # After saving, reload interviews from DB to ensure session state reflects new data
        st.session_state.interviews = load_interviews_from_db()

    except Exception as e:
        st.error(f"Error in evaluation: {str(e)}")
        logging.error(f"Evaluation error: {e}")
        st.session_state.interview_started_processing = False
        st.session_state.interview_processed_successfully = False

def recruiter_login_logic():
    st.subheader("🔑 Recruiter Login")
    password = st.text_input("Password", type="password", key="recruiter_password")
    if st.button("Login", key="recruiter_login_btn"):
        if password == RECRUITER_PASSWORD: # Using RECRUITER_PASSWORD from env
            st.session_state.current_page = "recruiter_dashboard"
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")

# --------- Initialize session state -----------

for key, default_value in {
    'current_page': "verification",
    'interview_data': {},
    'shortlisted_df': None,
    'interviews': {}, # This will be populated from DB on app start/rerun
    'current_question_index': 0,
    'interview_started_processing': False,
    'interview_processed_successfully': False,
    'authenticated': False,
    'dynamic_questions': [],
    'audio_question_played': False,
    'audio_bytes': None,
    'transcribed_text': ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Initial load of interviews from DB when app starts or refreshes ---
# This ensures that 'interviews' in session state always reflects the database content
if 'interviews' not in st.session_state or not st.session_state.interviews: # Load only if not loaded or empty
    st.session_state.interviews = load_interviews_from_db()


# --------- MAIN APP LOGIC ------------

st.title("AI Interview Portal 🚀")
st.markdown("---")

if st.session_state.current_page == "verification":
    st.header("📝 Candidate Verification")
    with st.expander("Candidate Information", expanded=True):
        full_name = st.text_input("Full Name")
        uploaded_file = st.file_uploader("Upload Resume/CV", type=["pdf", "doc", "docx", "txt"])

    st.markdown("---")
    st.subheader("Recruiter Actions")
    uploaded_excel = st.file_uploader("Upload Shortlisted Candidates (Excel)", type=["xlsx"])
    
    if uploaded_excel and st.button("Load Shortlisted List"):
        st.session_state.shortlisted_df = load_shortlisted_candidates_from_excel(uploaded_excel)
        if st.session_state.shortlisted_df is not None:
            st.success("Candidates loaded!")
            st.dataframe(st.session_state.shortlisted_df.head())

    st.markdown("---")
    if st.button("Recruiter Login"):
        st.session_state.current_page = "recruiter_login"
        st.rerun()

    if st.button("Start Interview", type="primary"):
        if not full_name.strip():
            st.error("Please provide your name")
        elif st.session_state.shortlisted_df is None:
            st.error("Shortlist not loaded")
        else:
            candidate_row = st.session_state.shortlisted_df[st.session_state.shortlisted_df['Name'].str.strip().str.lower() == full_name.strip().lower()]
            
            if candidate_row.empty:
                st.error("Name not found in shortlist")
            else:
                if uploaded_file:
                    verification_text = extract_text_from_document(uploaded_file)
                    if verification_text.strip():
                        candidate_jd = candidate_row['Job Description'].iloc[0] if 'Job Description' in candidate_row.columns else ""
                        
                        st.session_state.dynamic_questions = asyncio.run(generate_interview_questions(candidate_jd))
                        if not st.session_state.dynamic_questions:
                            st.session_state.dynamic_questions = [
                                "Tell me about yourself.",
                                "Describe your experience.",
                                "Why are you interested in this role?"
                            ]

                        st.session_state.current_page = "interview"
                        st.session_state.interview_data = {
                            "candidate_name": full_name.strip(),
                            "jd": candidate_jd.strip(),
                            "verification_text": verification_text.strip(),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.interview_data["qa"] = []
                        st.session_state.current_question_index = 0
                        st.session_state.audio_question_played = False
                        st.session_state.interview_processed_successfully = False
                        st.rerun()
                    else:
                        st.error("Invalid resume file")
                else:
                    st.error("Please upload your resume")

elif st.session_state.current_page == "interview":
    candidate_name = st.session_state.interview_data["candidate_name"]
    st.header(f"Interview: {candidate_name}")
    st.markdown("---")

    if st.session_state.interview_processed_successfully:
        st.subheader("✅ Interview Completed")
        st.markdown(f"**Score:** {st.session_state.interview_data['total_score']}/30")
        
        # Display feedback from the database (via reloaded session state)
        # Ensure we are looking at the current candidate's data from the loaded interviews
        current_candidate_interview_data = st.session_state.interviews.get(candidate_name, {})
        for i, qa in enumerate(current_candidate_interview_data.get("qa", []), 1):
            with st.expander(f"Question {i}"):
                st.write(f"**Q:** {qa['question']}")
                st.write(f"**A:** {qa['answer']}")
                if qa.get('audio_bytes'):
                    # Seek to start for playback
                    qa['audio_bytes'].seek(0)
                    st.audio(qa['audio_bytes'], format="audio/wav")
                st.markdown(f"**Score:** {qa.get('score', 'N/A')}/10")
                st.markdown(f"**Feedback:** {qa.get('feedback', 'None')}")

        if st.button("Back to Start"):
            st.session_state.current_page = "verification"
            # Clear current interview data for the next candidate
            st.session_state.interview_data = {}
            st.rerun()

    else:
        if st.session_state.current_question_index < len(st.session_state.dynamic_questions):
            current_question = st.session_state.dynamic_questions[st.session_state.current_question_index]
            st.subheader(f"Question {st.session_state.current_question_index + 1}/{len(st.session_state.dynamic_questions)}")
            
            if not st.session_state.audio_question_played:
                audio_bytes = text_to_speech(current_question)
                if audio_bytes:
                    st.write(f"**{current_question}**")
                    autoplay_audio(audio_bytes)
                st.session_state.audio_question_played = True
            else:
                st.write(f"**{current_question}**")
            
            st.write("Record your answer:")
            transcribed_text = record_audio()
            
            answer = st.text_area("Or type your answer", key=f"answer_{st.session_state.current_question_index}")
            
            final_answer = transcribed_text if transcribed_text else answer

            if st.button("Submit Answer"):
                if final_answer.strip():
                    # Save BytesIO object directly for now, will convert to bytes for DB
                    audio_for_save = st.session_state.audio_bytes if st.session_state.audio_bytes else None

                    st.session_state.interview_data["qa"].append({
                        "question": current_question,
                        "answer": final_answer.strip(),
                        "audio_bytes": audio_for_save
                    })
                    # Reset audio state for next question
                    st.session_state.audio_bytes = None
                    st.session_state.transcribed_text = ""

                    st.session_state.current_question_index += 1
                    st.session_state.audio_question_played = False
                    st.rerun()
                else:
                    st.warning("Please provide an answer")

        elif not st.session_state.interview_started_processing:
            st.info("All questions answered. Processing results...")
            st.session_state.interview_started_processing = True
            # The conduct_interview function will now save to DB
            asyncio.run(conduct_interview(st.session_state.dynamic_questions, st.session_state.interview_data["verification_text"]))
            st.rerun()

elif st.session_state.current_page == "recruiter_login":
    recruiter_login_logic()
    if st.button("Back to Candidate Verification"):
        st.session_state.current_page = "verification"
        st.rerun()

elif st.session_state.current_page == "recruiter_dashboard":
    if not st.session_state.authenticated:
        st.session_state.current_page = "recruiter_login"
        st.rerun()
    else:
        st.header("📊 Recruiter Dashboard")
        
        # Ensure interviews are always reloaded fresh for the dashboard
        st.session_state.interviews = load_interviews_from_db()

        if st.session_state.interviews:
            st.subheader("Completed Interviews")
            for candidate, data in st.session_state.interviews.items():
                with st.expander(f"{candidate} - {data['total_score']}/30 - {data['timestamp']}"): # Changed score key
                    st.write(f"Date: {data['timestamp']}")
                    st.write(f"Job Description (Excerpt): {data['jd'][:200]}...")
                    st.write(f"Resume Text (Excerpt): {data['resume_text'][:200]}...")

                    if data.get('qa'):
                        for i, qa_item in enumerate(data['qa']):
                            st.markdown(f"**Question {i+1}:** {qa_item['question']}")
                            st.markdown(f"**Answer:** {qa_item['answer']}")
                            if qa_item.get('audio_bytes'):
                                # BytesIO object needs to be reset for playback in Streamlit's audio widget
                                qa_item['audio_bytes'].seek(0)
                                st.audio(qa_item['audio_bytes'], format="audio/wav")
                            st.markdown(f"**Score:** {qa_item.get('score', 'N/A')}/10")
                            st.markdown(f"**Feedback:** {qa_item.get('feedback', 'None')}")
                            st.markdown("---") # Separator for Q&A pairs
                    
                    st.download_button(
                        label="Download Transcript",
                        data=format_transcript_for_download(data),
                        file_name=f"{candidate}_interview_{data['timestamp'].replace(':', '-')}.txt"
                    )
        else:
            st.info("No interviews completed yet")
            
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.current_page = "verification"
            st.rerun()

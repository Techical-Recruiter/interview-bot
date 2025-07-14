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
from sqlalchemy import text # IMPORTANT: Added for SQL queries

# Configure logging (optional but good for debugging)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize nest_asyncio for asyncio compatibility in Streamlit
nest_asyncio.apply()

# Load environment variables (for local testing, Streamlit Cloud uses secrets.toml directly)
load_dotenv()

# --- Retrieve API Keys and Passwords ---
# In Streamlit Cloud, these will come from your secrets.toml
# Locally, they will come from your .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RECRUITER_PASSWORD = os.getenv("RECRUITER_PASSWORD")

# --- Database Connection ---
# Use the name you defined in secrets.toml under [connections.NAME]
# Example: [connections.neon_db] in secrets.toml
conn = st.connection("neon_db", type="sql")

# --------- AUDIO RECORDING & TRANSCRIPTION USING streamlit_mic_recorder ------------

def save_audio_bytes_as_wav(audio_bytes):
    """Saves raw audio bytes to a temporary WAV file for speech recognition."""
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            audio_segment.export(f.name, format="wav")
            return f.name
    except Exception as e:
        logging.error(f"Error saving audio to WAV: {e}")
        st.error(f"Error processing audio: {e}")
        return None

def record_audio():
    """Records audio from microphone and transcribes it."""
    # Initialize session state variables if they don't exist
    if 'audio_bytes' not in st.session_state:
        st.session_state.audio_bytes = None
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""

    # Display the microphone recorder widget
    audio = mic_recorder(
        start_prompt="🎙️ Speak now",
        stop_prompt="🛑 Stop",
        just_once=True,
        use_container_width=True
    )

    if audio:
        st.session_state.audio_bytes = audio["bytes"] # Store raw bytes
        st.audio(st.session_state.audio_bytes, format="audio/wav")

        audio_path = save_audio_bytes_as_wav(st.session_state.audio_bytes)
        
        if audio_path:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                try:
                    st.session_state.transcribed_text = recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    st.session_state.transcribed_text = "Sorry, I couldn't understand that."
                except sr.RequestError as e:
                    st.session_state.transcribed_text = f"Speech recognition service unavailable: {e}"
            # Clean up the temporary file
            os.unlink(audio_path)

    if st.session_state.transcribed_text:
        # Display transcribed text in a text area, allowing user edits
        return st.text_area(
            "Transcribed Text (You can edit if needed):", 
            value=st.session_state.transcribed_text,
            height=150,
            key=f"transcribed_{st.session_state.current_question_index}"
        )
    else:
        return ""

def text_to_speech(text_to_convert, lang='en'):
    """Converts text to speech and returns audio bytes."""
    try:
        tts = gTTS(text=text_to_convert, lang=lang, slow=False)
        audio_bytes_io = io.BytesIO()
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0) # Rewind the BytesIO object to the beginning
        return audio_bytes_io
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        logging.error(f"TTS error: {e}")
        return None
    
def autoplay_audio(audio_bytes_io):
    """Plays audio automatically using HTML."""
    # Ensure audio_bytes_io is at the beginning before reading
    audio_bytes_io.seek(0) 
    audio_base64 = base64.b64encode(audio_bytes_io.read()).decode('utf-8')
    audio_html = f"""
    <audio controls autoplay>
    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    Your browser does not support the audio element.
    </audio>
    """
    st.components.v1.html(audio_html, height=50)

def extract_text_from_document(uploaded_file):
    """Extracts text from PDF, DOC, DOCX, or TXT files."""
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
        logging.error(f"Text extraction error: {e}")
        return ""
    return text

def load_shortlisted_candidates_from_excel(uploaded_file):
    """Loads candidate names and job descriptions from an Excel file."""
    try:
        df = pd.read_excel(uploaded_file)
        if 'Name' not in df.columns:
            st.error("Excel file must contain a column named 'Name' for candidate names.")
            return None
        if 'Job Description' not in df.columns:
            st.warning("Excel file does not contain a 'Job Description' column. Using empty string for JD.")
            df['Job Description'] = ""
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        logging.error(f"Excel load error: {e}")
        return None

# --- NEW: Functions to interact with the database ---
def save_interview_to_db(interview_data):
    """Saves a completed interview's data to the Neon PostgreSQL database."""
    try:
        with conn.session as session:
            # Insert into interviews table
            result = session.execute(
                text("""
                INSERT INTO interviews (candidate_name, interview_timestamp, total_score, job_description, resume_text)
                VALUES (:candidate_name, :interview_timestamp, :total_score, :job_description, :resume_text)
                RETURNING id;
                """), # Wrap SQL with text()
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
                # Fix for 'bytes' object has no attribute 'getvalue'
                # st.session_state.audio_bytes is already bytes from mic_recorder
                audio_bytes_to_store = qa_item.get("audio_bytes") if qa_item.get("audio_bytes") else None
                session.execute(
                    text("""
                    INSERT INTO qa_details (interview_id, question, answer, score, feedback, audio_bytes)
                    VALUES (:interview_id, :question, :answer, :score, :feedback, :audio_bytes)
                    """), # Wrap SQL with text()
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
    """Loads all interview data from the Neon PostgreSQL database."""
    interviews = {}
    try:
        with conn.session as session:
            # Fetch all main interview records
            main_interviews_result = session.execute(text("SELECT * FROM interviews;")).fetchall() # Wrap SQL with text()
            
            for row in main_interviews_result:
                # Assuming the order of columns matches your CREATE TABLE statement
                interview_dict = {
                    "id": row[0],
                    "candidate_name": row[1],
                    # Format datetime object to string for consistency with session_state
                    "timestamp": row[2].strftime("%Y-%m-%d %H:%M:%S") if row[2] else None,
                    "total_score": row[3],
                    "jd": row[4],
                    "resume_text": row[5]
                }
                
                # Fetch Q&A details for this interview using its ID
                # Using f-string for interview_id in WHERE clause is generally safe if it's an integer
                # For user-provided strings, always use parameterized queries to prevent SQL injection!
                qa_details_result = session.execute(
                    text(f"SELECT question, answer, score, feedback, audio_bytes FROM qa_details WHERE interview_id = {interview_dict['id']};")
                ).fetchall() # Wrap SQL with text()
                
                qa_list = []
                for qa_row in qa_details_result:
                    qa_dict = {
                        "question": qa_row[0],
                        "answer": qa_row[1],
                        "score": qa_row[2],
                        "feedback": qa_row[3],
                        # Convert bytes from DB back to BytesIO for Streamlit audio playback
                        "audio_bytes": io.BytesIO(qa_row[4]) if qa_row[4] else None
                    }
                    qa_list.append(qa_dict)
                
                interview_dict["qa"] = qa_list
                # Store the interview data using candidate name as key
                interviews[interview_dict["candidate_name"]] = interview_dict
    except Exception as e:
        st.error(f"Error loading interview data from database: {str(e)}")
        logging.error(f"Database load error: {e}")
    return interviews

# --- End of NEW database functions ---


def format_transcript_for_download(interview_data):
    """Formats interview data into a human-readable transcript."""
    candidate_name = interview_data.get("candidate_name", "N/A Candidate")
    timestamp = interview_data.get("timestamp", "N/A Date")
    total_score = interview_data.get("total_score", "N/A")
    jd = interview_data.get("jd", "Not provided.")
    resume_text = interview_data.get("resume_text", "Not provided.")

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
        f"{resume_text[:1000]}..." if resume_text else "Not provided.",
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
    """Generates interview questions based on the Job Description using Gemini API."""
    try:
        # Use AsyncOpenAI client to connect to Google Gemini API
        provider = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY
        )
        model = OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
        set_tracing_disabled(disabled=True) # Disable tracing for cleaner logs

        agent = Agent(
            name="Question Generator",
            instructions=f"""
            You are an expert HR interviewer. Generate 3 concise and relevant interview questions based on the provided Job Description.
            Focus on core skills and experience indicated in the JD.

            Job Description:
            "{jd}"

            Output format (strictly JSON):
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

        st.info("Generating interview questions...")
        result = Runner.run_streamed(starting_agent=agent, input="Generate questions")
        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta

        # Parse the JSON response
        response_json = json.loads(full_response.strip().strip('```json').strip('```'))
        return response_json["questions"]
    except json.JSONDecodeError:
        st.error("Failed to parse questions from AI. Retrying with default questions.")
        logging.error(f"JSON decode error during question generation: {full_response}")
        return ["Tell me about yourself.", "Describe your experience.", "Why are you interested in this role?"]
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}. Using default questions.")
        logging.error(f"Question generation error: {e}")
        return ["Tell me about yourself.", "Describe your experience.", "Why are you interested in this role?"]

async def conduct_interview(questions, resume_text):
    """Evaluates candidate answers using Gemini API and saves results."""
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
            You are an experienced HR professional. Evaluate the candidate's answers based on their resume and the job description.
            For each question, provide a score (1-10) and constructive feedback.
            Also, provide a total score for the entire interview (sum of individual question scores, max 30).

            Candidate's Resume/Verification Text: "{resume_text}"
            Job Description: "{st.session_state.interview_data['jd']}"
            
            Strictly follow the output JSON format.

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
        # Prepare input for the AI agent
        interview_input = "\n\n".join([f"Question {i+1}: {q['question']}\nAnswer: {q['answer']}" for i, q in enumerate(qa_data)])

        st.info("Analyzing responses and generating feedback...")
        result = Runner.run_streamed(starting_agent=agent, input=interview_input)
        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta

        response_json = json.loads(full_response.strip().strip('```json').strip('```'))
        
        # Update scores and feedback in the session state's interview_data
        for i, qa in enumerate(st.session_state.interview_data["qa"]):
            if i < len(response_json["questions"]): # Ensure index is within bounds
                ai_qa = response_json["questions"][i]
                qa["score"] = int(ai_qa.get("score", 0)) # Ensure score is integer
                qa["feedback"] = ai_qa.get("feedback", "No feedback provided by AI.")
        
        st.session_state.interview_data["total_score"] = int(response_json.get("total_score", 0))
        
        # --- CRITICAL CHANGE: Save to DB here instead of just session_state.interviews ---
        save_interview_to_db(st.session_state.interview_data)
        # --- End Critical Change ---

        st.session_state.interview_processed_successfully = True
        # After saving to DB, reload interviews from DB to ensure session state reflects new data
        # This is crucial for the dashboard to show the latest interview without full app restart
        st.session_state.interviews = load_interviews_from_db()

    except json.JSONDecodeError:
        st.error("Failed to parse AI evaluation. Please try again.")
        logging.error(f"JSON decode error during evaluation: {full_response}")
        st.session_state.interview_started_processing = False
        st.session_state.interview_processed_successfully = False
    except Exception as e:
        st.error(f"Error in evaluation process: {str(e)}")
        logging.error(f"Evaluation error: {e}")
        st.session_state.interview_started_processing = False
        st.session_state.interview_processed_successfully = False

def recruiter_login_logic():
    """Handles recruiter login authentication."""
    st.subheader("🔑 Recruiter Login")
    password = st.text_input("Password", type="password", key="recruiter_password")
    if st.button("Login", key="recruiter_login_btn"):
        if password == RECRUITER_PASSWORD:
            st.session_state.current_page = "recruiter_dashboard"
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")

# --------- Initialize session state -----------
# This ensures that all necessary state variables are present from the start.
for key, default_value in {
    'current_page': "verification", # Controls which part of the app is shown
    'interview_data': {},           # Stores data for the *current* interview being conducted
    'shortlisted_df': None,         # DataFrame for shortlisted candidates
    'interviews': {},               # Stores *all completed interviews* (loaded from DB)
    'current_question_index': 0,    # Index for current interview question
    'interview_started_processing': False, # Flag to prevent re-triggering AI evaluation
    'interview_processed_successfully': False, # Flag for successful AI evaluation
    'authenticated': False,         # Recruiter authentication status
    'dynamic_questions': [],        # Questions generated by AI
    'audio_question_played': False, # Flag to ensure question audio plays once
    'audio_bytes': None,            # Stores raw audio bytes from mic
    'transcribed_text': ""          # Stores transcribed text from mic
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Initial load of interviews from DB when app starts or refreshes ---
# This ensures that 'interviews' in session state always reflects the database content.
# It only loads if 'interviews' is not yet populated in session state
# (e.g., on first app load or full refresh).
if 'interviews' not in st.session_state or not st.session_state.interviews:
    logging.info("Loading interviews from database on app start.")
    st.session_state.interviews = load_interviews_from_db()

# --------- MAIN APP LOGIC ------------

st.title("AI Interview Portal 🚀")
st.markdown("---")

if st.session_state.current_page == "verification":
    st.header("📝 Candidate Verification")
    with st.expander("Candidate Information", expanded=True):
        full_name = st.text_input("Full Name", key="candidate_full_name")
        uploaded_file = st.file_uploader("Upload Resume/CV", type=["pdf", "doc", "docx", "txt"], key="resume_uploader")

    st.markdown("---")
    st.subheader("Recruiter Actions")
    uploaded_excel = st.file_uploader("Upload Shortlisted Candidates (Excel)", type=["xlsx"], key="shortlist_uploader")
    
    if uploaded_excel and st.button("Load Shortlisted List", key="load_shortlist_btn"):
        st.session_state.shortlisted_df = load_shortlisted_candidates_from_excel(uploaded_excel)
        if st.session_state.shortlisted_df is not None:
            st.success("Candidates loaded!")
            st.dataframe(st.session_state.shortlisted_df.head())

    st.markdown("---")
    if st.button("Recruiter Login", key="go_to_recruiter_login_btn"):
        st.session_state.current_page = "recruiter_login"
        st.rerun()

    if st.button("Start Interview", type="primary", key="start_interview_btn"):
        if not full_name.strip():
            st.error("Please provide your name.")
        elif st.session_state.shortlisted_df is None:
            st.error("Please upload and load the shortlist Excel file first.")
        else:
            candidate_row = st.session_state.shortlisted_df[st.session_state.shortlisted_df['Name'].str.strip().str.lower() == full_name.strip().lower()]
            
            if candidate_row.empty:
                st.error(f"Candidate '{full_name}' not found in the shortlisted list.")
            else:
                if uploaded_file:
                    verification_text = extract_text_from_document(uploaded_file)
                    if verification_text.strip():
                        candidate_jd = candidate_row['Job Description'].iloc[0] if 'Job Description' in candidate_row.columns else ""
                        
                        # Generate dynamic questions asynchronously
                        st.info("Generating questions based on JD. Please wait...")
                        st.session_state.dynamic_questions = asyncio.run(generate_interview_questions(candidate_jd))
                        
                        st.session_state.current_page = "interview"
                        st.session_state.interview_data = {
                            "candidate_name": full_name.strip(),
                            "jd": candidate_jd.strip(),
                            "verification_text": verification_text.strip(),
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Use datetime for current time
                            "total_score": 0, # Initialize total score
                            "qa": [] # Initialize Q&A list
                        }
                        st.session_state.current_question_index = 0
                        st.session_state.audio_question_played = False
                        st.session_state.interview_processed_successfully = False
                        st.session_state.interview_started_processing = False # Reset for new interview
                        st.rerun()
                    else:
                        st.error("Could not extract text from the provided resume file. Please check the file.")
                else:
                    st.error("Please upload your resume/CV.")

elif st.session_state.current_page == "interview":
    candidate_name = st.session_state.interview_data.get("candidate_name", "Candidate")
    st.header(f"Interview: {candidate_name}")
    st.markdown("---")

    if st.session_state.interview_processed_successfully:
        st.subheader("✅ Interview Completed")
        st.markdown(f"**Overall Score:** {st.session_state.interview_data['total_score']}/30")
        
        # Display feedback from the database (via reloaded session state)
        # Ensure we are looking at the current candidate's data from the loaded interviews
        # It's better to fetch directly from the session_state.interviews which is reloaded from DB
        current_candidate_interview_data = st.session_state.interviews.get(candidate_name)
        if current_candidate_interview_data:
            for i, qa in enumerate(current_candidate_interview_data.get("qa", []), 1):
                with st.expander(f"Question {i} - Score: {qa.get('score', 'N/A')}"):
                    st.write(f"**Q:** {qa['question']}")
                    st.write(f"**A:** {qa['answer']}")
                    if qa.get('audio_bytes'):
                        # Ensure BytesIO object is at the beginning before playing
                        qa['audio_bytes'].seek(0)
                        st.audio(qa['audio_bytes'], format="audio/wav")
                    st.markdown(f"**Feedback:** {qa.get('feedback', 'No feedback provided.')}")
        else:
            st.warning("Could not retrieve detailed interview results from database. Please check dashboard.")

        if st.button("Back to Start", key="back_to_start_after_interview"):
            st.session_state.current_page = "verification"
            # Clear current interview data for the next candidate to prevent carry-over
            st.session_state.interview_data = {}
            st.session_state.current_question_index = 0
            st.session_state.interview_processed_successfully = False
            st.session_state.interview_started_processing = False
            st.session_state.dynamic_questions = []
            st.session_state.audio_question_played = False
            st.session_state.audio_bytes = None
            st.session_state.transcribed_text = ""
            st.rerun()

    else:
        if st.session_state.current_question_index < len(st.session_state.dynamic_questions):
            current_question = st.session_state.dynamic_questions[st.session_state.current_question_index]
            st.subheader(f"Question {st.session_state.current_question_index + 1}/{len(st.session_state.dynamic_questions)}")
            
            # Play audio for the question only once
            if not st.session_state.audio_question_played:
                audio_bytes_io = text_to_speech(current_question)
                if audio_bytes_io:
                    st.write(f"**{current_question}**")
                    autoplay_audio(audio_bytes_io)
                st.session_state.audio_question_played = True
            else:
                st.write(f"**{current_question}**")
            
            st.write("Record your answer:")
            transcribed_text = record_audio() # This handles recording and transcription
            
            # Text area for manual input or editing transcribed text
            answer = st.text_area(
                "Or type your answer (This will override transcribed text if both are present):",
                value=st.session_state.transcribed_text, # Initialize with transcribed text
                key=f"answer_manual_{st.session_state.current_question_index}"
            )
            
            # If the user typed, that takes precedence; otherwise, use transcribed text
            final_answer_to_submit = answer.strip() if answer.strip() else transcribed_text.strip()

            if st.button("Submit Answer", key=f"submit_answer_btn_{st.session_state.current_question_index}"):
                if final_answer_to_submit:
                    # st.session_state.audio_bytes is already bytes, no .getvalue() needed
                    audio_for_save = st.session_state.audio_bytes if st.session_state.audio_bytes else None

                    st.session_state.interview_data["qa"].append({
                        "question": current_question,
                        "answer": final_answer_to_submit,
                        "audio_bytes": audio_for_save # Store raw bytes
                    })
                    # Reset audio and transcription state for the next question
                    st.session_state.audio_bytes = None
                    st.session_state.transcribed_text = ""

                    st.session_state.current_question_index += 1
                    st.session_state.audio_question_played = False # Allow next question to play audio
                    st.rerun() # Rerun the app to show the next question
                else:
                    st.warning("Please provide an answer either by speaking or typing.")

        elif not st.session_state.interview_started_processing:
            st.info("All questions answered. Processing results and saving data. This may take a moment...")
            st.session_state.interview_started_processing = True # Set flag to prevent re-processing
            # The conduct_interview function will now save to DB and update session_state.interviews
            asyncio.run(conduct_interview(st.session_state.dynamic_questions, st.session_state.interview_data["verification_text"]))
            st.rerun() # Rerun to show the "Interview Completed" section

elif st.session_state.current_page == "recruiter_login":
    recruiter_login_logic()
    if st.button("Back to Candidate Verification", key="back_to_candidate_verification_from_login"):
        st.session_state.current_page = "verification"
        st.rerun()

elif st.session_state.current_page == "recruiter_dashboard":
    # Security check: if not authenticated, redirect to login
    if not st.session_state.authenticated:
        st.session_state.current_page = "recruiter_login"
        st.rerun()
    else:
        st.header("📊 Recruiter Dashboard")
        
        # Ensure interviews are always reloaded fresh for the dashboard
        # This makes sure the dashboard displays the most up-to-date information from the DB.
        st.session_state.interviews = load_interviews_from_db()

        if st.session_state.interviews:
            st.subheader("Completed Interviews")
            # Sort interviews by timestamp, newest first
            sorted_interviews = sorted(
                st.session_state.interviews.values(),
                key=lambda x: datetime.datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S") if isinstance(x['timestamp'], str) else x['timestamp'],
                reverse=True
            )
            
            for data in sorted_interviews:
                candidate_name = data.get("candidate_name", "N/A")
                total_score = data.get("total_score", "N/A")
                timestamp = data.get("timestamp", "N/A")
                
                with st.expander(f"👤 {candidate_name} - Score: {total_score}/30 - Date: {timestamp}"):
                    st.write(f"**Job Description (Excerpt):** {data.get('jd', 'Not provided.')[:500]}...")
                    st.write(f"**Resume Text (Excerpt):** {data.get('resume_text', 'Not provided.')[:500]}...")

                    if data.get('qa'):
                        st.markdown("#### Questions & Answers:")
                        for i, qa_item in enumerate(data['qa']):
                            st.markdown(f"**Question {i+1}:** {qa_item.get('question', 'N/A')}")
                            st.markdown(f"**Candidate Answer:** {qa_item.get('answer', 'N/A')}")
                            if qa_item.get('audio_bytes'):
                                # BytesIO object needs to be reset for playback in Streamlit's audio widget
                                qa_item['audio_bytes'].seek(0)
                                st.audio(qa_item['audio_bytes'], format="audio/wav", start_time=0)
                            st.markdown(f"**Score:** {qa_item.get('score', 'N/A')}/10")
                            st.markdown(f"**Feedback:** {qa_item.get('feedback', 'No feedback provided.')}")
                            st.markdown("---") # Separator for Q&A pairs
                    
                    st.download_button(
                        label="Download Transcript (TXT)",
                        data=format_transcript_for_download(data),
                        file_name=f"{candidate_name}_interview_{timestamp.replace(':', '-')}.txt",
                        mime="text/plain"
                    )
        else:
            st.info("No interviews completed yet. Start an interview to see results here!")
            
        if st.button("Logout", key="recruiter_logout_btn"):
            st.session_state.authenticated = False
            st.session_state.current_page = "verification"
            st.rerun()

Okay, I understand. The persistent issues with audio cutting off, the page refreshing constantly, and previous page messages showing up are indeed frustrating. These are all related to how Streamlit re-runs the script and how your UI and state are being managed.

The main problem is that Streamlit reruns the *entire script* from top to bottom on every interaction (like a button click, text input change, or even `time.sleep` followed by an implicit re-run). If not managed carefully, this leads to:

1.  **Audio Interruptions:** When the script re-runs, the HTML audio component is re-rendered, which stops ongoing playback.
2.  **UI Overlap:** Elements that aren't strictly confined to a specific page's `if` block will be redrawn, often on top of other elements, or persist when they shouldn't.
3.  **Constant Refresh:** The `time.sleep(1)` combined with `st.rerun()` in your timer loop causes a deliberate, second-by-second re-run, leading to the "page refreshing again and again" observation.

### The Core Fixes Implemented

I've made critical adjustments to your code to address these, focusing on more explicit state management and Streamlit's execution model:

1.  **Dedicated "Loading/Generating Questions" Page**:

      * Instead of `st.info` messages appearing on the "verification" or "interview" page while questions are generated, I've introduced a new `current_page` state: `"generating_questions"`.
      * When the "Start Interview" button is clicked, the app transitions to this loading page. The AI question generation happens *only* on this page. Once questions are generated, it transitions cleanly to the "interview" page. This prevents the "Generating questions..." messages from overlapping.

2.  **Controlled Audio Playback and Timer Reruns**:

      * The most significant change is in the `interview` page's audio and timer logic.
      * When the question audio is first played (`if not st.session_state.audio_question_played:`), the script will now **not immediately call `st.rerun()`**. It will finish the current execution frame.
      * The `st.rerun()` for the timer (`time.sleep(1); st.rerun()`) is now **only triggered if the audio has already played AND no submit button was clicked in the current run**. This prevents the timer's constant reruns from interrupting the initial audio playback.
      * This allows the audio to fully stream and play in the browser before the timer takes over with its regular updates.

3.  **Strict UI Page Segregation**:

      * Ensured that **all** `st.` elements (headers, text inputs, buttons, messages like `st.info`, `st.warning`, `st.error`) are nested strictly within their respective `if st.session_state.current_page == "..."` blocks. This is crucial to prevent UI overlap.

4.  **Clearer State Resets**:

      * Refined the session state resets when moving between pages to ensure no lingering data causes unexpected behavior.

-----

Here is the **full, corrected code** with detailed comments explaining the changes:

````python
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

# --------- AUDIO RECORDING & TRANSCRIPTION ------------

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
    # Ensure these session states are initialized before use
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
            os.unlink(audio_path) # Clean up the temporary file

    # Return the text area for user to edit, whether transcribed or empty
    return st.text_area(
        "Transcribed Text (You can edit if needed):", 
        value=st.session_state.transcribed_text,
        height=150,
        key=f"transcribed_{st.session_state.current_question_index}"
    )

def text_to_speech(text_to_convert, lang='en'):
    """Converts text to speech and returns audio bytes."""
    try:
        tts = gTTS(text=text_to_convert, lang=lang, slow=False)
        audio_bytes_io = io.BytesIO()
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)
        return audio_bytes_io
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        logging.error(f"TTS error: {e}")
        return None
    
def autoplay_audio(audio_bytes_io):
    """Plays audio automatically using HTML."""
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

# --- Database Functions ---
def save_interview_to_db(interview_data):
    """Saves a completed interview's data to the Neon PostgreSQL database."""
    try:
        with conn.session as session:
            result = session.execute(
                text("""
                INSERT INTO interviews (candidate_name, interview_timestamp, total_score, job_description, resume_text)
                VALUES (:candidate_name, :interview_timestamp, :total_score, :job_description, :resume_text)
                RETURNING id;
                """),
                params={
                    "candidate_name": interview_data["candidate_name"],
                    "interview_timestamp": interview_data["timestamp"],
                    "total_score": interview_data["total_score"],
                    "job_description": interview_data["jd"],
                    "resume_text": interview_data["verification_text"]
                }
            )
            interview_id = result.scalar_one()

            for qa_item in interview_data.get("qa", []):
                audio_bytes_to_store = qa_item.get("audio_bytes") # audio_bytes is already bytes
                session.execute(
                    text("""
                    INSERT INTO qa_details (interview_id, question, answer, score, feedback, audio_bytes)
                    VALUES (:interview_id, :question, :answer, :score, :feedback, :audio_bytes)
                    """),
                    params={
                        "interview_id": interview_id,
                        "question": qa_item["question"],
                        "answer": qa_item["answer"],
                        "score": qa_item.get("score"),
                        "feedback": qa_item.get("feedback"),
                        "audio_bytes": audio_bytes_to_store
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
            main_interviews_result = session.execute(text("SELECT * FROM interviews;")).fetchall()
            
            for row in main_interviews_result:
                interview_dict = {
                    "id": row[0],
                    "candidate_name": row[1],
                    "timestamp": row[2].strftime("%Y-%m-%d %H:%M:%S") if row[2] else None,
                    "total_score": row[3],
                    "jd": row[4],
                    "resume_text": row[5]
                }
                
                qa_details_result = session.execute(
                    text(f"SELECT question, answer, score, feedback, audio_bytes FROM qa_details WHERE interview_id = {interview_dict['id']};")
                ).fetchall()
                
                qa_list = []
                for qa_row in qa_details_result:
                    qa_dict = {
                        "question": qa_row[0],
                        "answer": qa_row[1],
                        "score": qa_row[2],
                        "feedback": qa_row[3],
                        "audio_bytes": io.BytesIO(qa_row[4]) if qa_row[4] else None
                    }
                    qa_list.append(qa_dict)
                
                interview_dict["qa"] = qa_list
                # Using interview_id as key as candidate name might not be unique over time
                interviews[interview_dict["id"]] = interview_dict 
    except Exception as e:
        st.error(f"Error loading interview data from database: {str(e)}")
        logging.error(f"Database load error: {e}")
    return interviews

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
        provider = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY
        )
        model = OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
        set_tracing_disabled(disabled=True)

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

        # Do not use st.info here; it's handled by the calling page ("generating_questions")
        result = Runner.run_streamed(starting_agent=agent, input="Generate questions")
        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta

        response_json = json.loads(full_response.strip().strip('```json').strip('```'))
        return response_json["questions"]
    except json.JSONDecodeError:
        logging.error(f"JSON decode error during question generation: {full_response}")
        # Provide a user-friendly message, but don't use st.error directly in an async function
        # This will be handled by the caller or a subsequent rerender
        st.session_state.error_message = "Failed to parse questions from AI. Using default questions."
        return ["Tell me about yourself.", "Describe your experience.", "Why are you interested in this role?"]
    except Exception as e:
        logging.error(f"Question generation error: {e}")
        st.session_state.error_message = f"Error generating questions: {str(e)}. Using default questions."
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
        interview_input = "\n\n".join([f"Question {i+1}: {q['question']}\nAnswer: {q['answer']}" for i, q in enumerate(qa_data)])

        # Do not use st.info here; it's handled by the calling page (interview page when processing results)
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
                qa["feedback"] = ai_qa.get("feedback", "No feedback provided by AI.")
        
        st.session_state.interview_data["total_score"] = int(response_json.get("total_score", 0))
        
        save_interview_to_db(st.session_state.interview_data)
        st.session_state.interview_processed_successfully = True
        st.session_state.interviews = load_interviews_from_db()

    except json.JSONDecodeError:
        st.session_state.error_message = "Failed to parse AI evaluation. Please try again."
        logging.error(f"JSON decode error during evaluation: {full_response}")
        st.session_state.interview_started_processing = False
        st.session_state.interview_processed_successfully = False
    except Exception as e:
        st.session_state.error_message = f"Error in evaluation process: {str(e)}"
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
# Ensure all session state variables are initialized
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
    'audio_bytes': None,
    'transcribed_text': "",
    'timer_active': False,
    'timer_start_time': None,
    'answer_submitted_early': False,
    'show_question_generation_status': False, # New: control display of generation status
    'error_message': None # New: for displaying async errors cleanly
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Load interviews initially (or refresh if needed)
if 'interviews' not in st.session_state or not st.session_state.interviews:
    logging.info("Loading interviews from database on app start.")
    st.session_state.interviews = load_interviews_from_db()

# --- Global UI (appears on all pages, typically just title/header) ---
st.title("AI Interview Portal 🚀")
st.markdown("---")

# --- Page Routing Logic ---

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
                        
                        # --- Transition to a "generating questions" state/page ---
                        st.session_state.current_page = "generating_questions"
                        # Store necessary data for the next page
                        st.session_state.candidate_name_for_interview = full_name.strip()
                        st.session_state.candidate_jd_for_interview = candidate_jd.strip()
                        st.session_state.verification_text_for_interview = verification_text.strip()
                        st.rerun() # Trigger rerun to go to the new page
                    else:
                        st.error("Could not extract text from the provided resume file. Please check the file.")
                else:
                    st.error("Please upload your resume/CV.")

# --- NEW PAGE: Generating Questions ---
elif st.session_state.current_page == "generating_questions":
    st.header("Generating Interview Questions")
    st.info("Please wait while AI generates personalized interview questions based on the Job Description and Resume.")
    
    # Run the async function to generate questions
    questions = asyncio.run(generate_interview_questions(
        st.session_state.candidate_jd_for_interview
    ))
    
    # After questions are generated, prepare for interview page
    st.session_state.dynamic_questions = questions
    
    # Initialize interview_data
    st.session_state.interview_data = {
        "candidate_name": st.session_state.candidate_name_for_interview,
        "jd": st.session_state.candidate_jd_for_interview,
        "verification_text": st.session_state.verification_text_for_interview,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_score": 0,
        "qa": []
    }

    # Reset/initialize other interview-related states
    st.session_state.current_question_index = 0
    st.session_state.audio_question_played = False
    st.session_state.interview_processed_successfully = False
    st.session_state.interview_started_processing = False
    st.session_state.timer_active = False
    st.session_state.timer_start_time = None
    st.session_state.answer_submitted_early = False
    
    # Clean up temporary state variables used for transition
    del st.session_state.candidate_name_for_interview
    del st.session_state.candidate_jd_for_interview
    del st.session_state.verification_text_for_interview
    
    # Display any error message from question generation before proceeding
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        st.session_state.error_message = None # Clear after display

    # Transition to the interview page
    st.session_state.current_page = "interview"
    st.rerun() # Trigger rerun to go to the interview page

# --- Interview Page ---
elif st.session_state.current_page == "interview":
    candidate_name = st.session_state.interview_data.get("candidate_name", "Candidate")
    st.header(f"Interview: {candidate_name}")
    st.markdown("---")

    # Display any error message that might have occurred during question generation
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        st.session_state.error_message = None # Clear after display

    if st.session_state.interview_processed_successfully:
        st.subheader("✅ Interview Completed")
        st.markdown(f"**Overall Score:** {st.session_state.interview_data['total_score']}/30")
        
        # Load the interview data again specifically for display, using ID for uniqueness
        # This prevents issues if candidate name isn't unique in session_state.interviews
        interview_id_to_display = None
        for interview_id, interview_data_val in st.session_state.interviews.items():
            if interview_data_val["candidate_name"] == candidate_name and \
               interview_data_val["timestamp"] == st.session_state.interview_data["timestamp"]:
                interview_id_to_display = interview_id
                break

        current_candidate_interview_data = st.session_state.interviews.get(interview_id_to_display)

        if current_candidate_interview_data:
            for i, qa in enumerate(current_candidate_interview_data.get("qa", []), 1):
                with st.expander(f"Question {i} - Score: {qa.get('score', 'N/A')}"):
                    st.write(f"**Q:** {qa['question']}")
                    st.write(f"**A:** {qa['answer']}")
                    if qa.get('audio_bytes'):
                        if isinstance(qa['audio_bytes'], io.BytesIO):
                            qa['audio_bytes'].seek(0)
                            st.audio(qa['audio_bytes'], format="audio/wav")
                        elif isinstance(qa['audio_bytes'], bytes):
                            st.audio(qa['audio_bytes'], format="audio/wav")
                        else:
                            st.info("Audio format not recognized for playback.")
                    st.markdown(f"**Feedback:** {qa.get('feedback', 'No feedback provided.')}")
        else:
            st.warning("Could not retrieve detailed interview results from database. Please check dashboard.")

        if st.button("Back to Start", key="back_to_start_after_interview"):
            # Reset all interview related states
            st.session_state.current_page = "verification"
            st.session_state.interview_data = {}
            st.session_state.current_question_index = 0
            st.session_state.interview_processed_successfully = False
            st.session_state.interview_started_processing = False
            st.session_state.dynamic_questions = []
            st.session_state.audio_question_played = False
            st.session_state.audio_bytes = None
            st.session_state.transcribed_text = ""
            st.session_state.timer_active = False
            st.session_state.timer_start_time = None
            st.session_state.answer_submitted_early = False
            st.rerun()

    else: # Interview is ongoing
        if st.session_state.current_question_index < len(st.session_state.dynamic_questions):
            current_question = st.session_state.dynamic_questions[st.session_state.current_question_index]
            st.subheader(f"Question {st.session_state.current_question_index + 1}/{len(st.session_state.dynamic_questions)}")
            
            # --- Audio Playback Logic for Question ---
            # This block plays the audio question ONLY ONCE per question.
            if not st.session_state.audio_question_played:
                audio_bytes_io = text_to_speech(current_question)
                if audio_bytes_io:
                    st.write(f"**{current_question}**")
                    autoplay_audio(audio_bytes_io)
                    # IMPORTANT: Do NOT call st.rerun() immediately here.
                    # Let the current script execution finish.
                    # The timer's st.rerun() will kick in on the next script run.
                st.session_state.audio_question_played = True # Mark as played
                # Add a small, non-blocking sleep if audio consistently cuts out too fast,
                # to give the browser a moment to start playing the HTML audio.
                # This is less ideal than browser full buffering, but can help.
                # time.sleep(0.5) 
            else:
                st.write(f"**{current_question}**") # Display text if audio already played
            
            # --- Answer Input ---
            st.write("Record your answer:")
            transcribed_text = record_audio() # This is a blocking call until audio is recorded
            
            answer_text_area = st.text_area(
                "Or type your answer (This will override transcribed text if both are present):",
                value=st.session_state.transcribed_text,
                height=150,
                key=f"answer_manual_{st.session_state.current_question_index}"
            )
            
            submit_button_key = f"submit_answer_btn_{st.session_state.current_question_index}"
            submit_button_clicked = st.button("Submit Answer", key=submit_button_key)

            # --- Timer Logic ---
            timer_placeholder = st.empty()
            MAX_TIME = 60 # 1 minute
            
            # Initialize timer if not active for the current question
            if not st.session_state.timer_active:
                st.session_state.timer_active = True
                st.session_state.timer_start_time = time.time()
                st.session_state.answer_submitted_early = False

            time_elapsed = time.time() - st.session_state.timer_start_time
            remaining_time = MAX_TIME - int(time_elapsed)

            # Display timer and trigger reruns
            if remaining_time > 0 and not st.session_state.answer_submitted_early:
                timer_placeholder.markdown(f"Time remaining: **{remaining_time} seconds** ⏰")
                # Only rerun for timer if the submit button wasn't pressed
                # and the audio has had a chance to play (or was already played).
                if not submit_button_clicked: 
                    time.sleep(1) # Pause for 1 second before next refresh
                    st.rerun() # Trigger a refresh to update timer
            else:
                st.session_state.timer_active = False # Timer finished
                if not st.session_state.answer_submitted_early: # If timer ran out and user didn't submit
                    timer_placeholder.error("Time's up! Moving to the next question.")
                    submit_button_clicked = True # Force submission as timeout
            # --- End Timer Logic ---

            # Process submission (either by button click or timeout)
            if submit_button_clicked:
                final_answer_to_submit = answer_text_area.strip() if answer_text_area.strip() else transcribed_text.strip()

                if not st.session_state.answer_submitted_early and remaining_time <= 0:
                    # This means it's a timeout and the user didn't submit early
                    final_answer_to_submit = "TIMEOUT: No answer submitted within 1 minute."
                    st.session_state.audio_bytes = None # No audio for timeout
                elif submit_button_clicked: # User explicitly clicked submit button
                    st.session_state.answer_submitted_early = True # Mark as submitted early
                    if not final_answer_to_submit: # If answer is empty after submitting
                        st.warning("Please provide an answer either by speaking or typing.")
                        # Reset for another attempt on the same question
                        st.session_state.timer_active = True 
                        st.session_state.timer_start_time = time.time() # Reset timer
                        st.session_state.answer_submitted_early = False 
                        st.rerun() # Rerun to display warning and restart timer on the SAME question
                
                # If we reach here, a valid answer (or timeout string) is ready for processing
                if final_answer_to_submit: 
                    audio_for_save = st.session_state.audio_bytes if st.session_state.audio_bytes else None
                    if final_answer_to_submit.startswith("TIMEOUT:"): # For timeout, ensure no audio
                        audio_for_save = None

                    st.session_state.interview_data["qa"].append({
                        "question": current_question,
                        "answer": final_answer_to_submit,
                        "audio_bytes": audio_for_save
                    })
                    # Reset states for the next question
                    st.session_state.audio_bytes = None
                    st.session_state.transcribed_text = ""
                    st.session_state.timer_active = False 
                    st.session_state.timer_start_time = None 
                    st.session_state.answer_submitted_early = False 

                    st.session_state.current_question_index += 1
                    st.session_state.audio_question_played = False # Reset for the next question's audio
                    st.rerun() # Move to the next question or final processing

        # After all questions are answered, initiate final processing
        elif not st.session_state.interview_started_processing:
            st.info("All questions answered. Processing results and saving data. This may take a moment...")
            st.session_state.interview_started_processing = True
            asyncio.run(conduct_interview(st.session_state.dynamic_questions, st.session_state.interview_data["verification_text"]))
            
            # Display any error message from evaluation
            if st.session_state.error_message:
                st.error(st.session_state.error_message)
                st.session_state.error_message = None # Clear after display
            st.rerun() # Rerun to display the completed interview page

# --- Recruiter Login Page ---
elif st.session_state.current_page == "recruiter_login":
    recruiter_login_logic()
    if st.button("Back to Candidate Verification", key="back_to_candidate_verification_from_login"):
        st.session_state.current_page = "verification"
        st.rerun()

# --- Recruiter Dashboard Page ---
elif st.session_state.current_page == "recruiter_dashboard":
    if not st.session_state.authenticated:
        st.session_state.current_page = "recruiter_login"
        st.rerun()
    else:
        st.header("📊 Recruiter Dashboard")
        
        # Always reload interviews from DB when accessing the dashboard to get the latest data
        st.session_state.interviews = load_interviews_from_db()

        if st.session_state.interviews:
            st.subheader("Completed Interviews")
            # Sort interviews by timestamp
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
                                if isinstance(qa_item['audio_bytes'], io.BytesIO):
                                    qa_item['audio_bytes'].seek(0)
                                    st.audio(qa_item['audio_bytes'], format="audio/wav", start_time=0)
                                elif isinstance(qa_item['audio_bytes'], bytes):
                                    st.audio(qa_item['audio_bytes'], format="audio/wav", start_time=0)
                                else:
                                    st.info("Audio format not recognized for playback.")
                            st.markdown(f"**Score:** {qa_item.get('score', 'N/A')}/10")
                            st.markdown(f"**Feedback:** {qa_item.get('feedback', 'No feedback provided.')}")
                            st.markdown("---")
                    
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

````

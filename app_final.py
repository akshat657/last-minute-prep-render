import streamlit as st
import cheatsheet_app
import yt_summary_app

st.set_page_config(
    page_title="AI Study Assistant", 
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS with Dark Mode Support
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Rubik', sans-serif;
        }
       
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .stApp {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            }
        }
       
        .title {
            font-size: 3. 5rem;
            font-weight: 700;
            color:  #1a1a2e;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        @media (prefers-color-scheme: dark) {
            .title {
                color: #ffffff;
                text-shadow: 2px 2px 8px rgba(108, 92, 231, 0.5);
            }
        }
        
        .subtitle {
            font-size: 1.3rem;
            color: #2d3436;
            text-align: center;
            margin-bottom: 1.5rem;
            font-weight: 500;
        }
        
        @media (prefers-color-scheme: dark) {
            .subtitle {
                color: #e0e0e0;
            }
        }
        
        . description {
            font-size: 1.1rem;
            color: #2d3436;
            text-align: center;
            line-height: 1.8;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        @media (prefers-color-scheme: dark) {
            .description {
                background: rgba(30, 30, 46, 0.9);
                color: #e0e0e0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.5);
            }
        }
        
        .feature-box {
            background: rgba(255, 255, 255, 0.9);
            padding: 1.2rem;
            border-radius:  10px;
            margin:  0.8rem 0;
            border-left: 4px solid #6c5ce7;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        @media (prefers-color-scheme: dark) {
            .feature-box {
                background: rgba(30, 30, 46, 0.9);
                box-shadow: 0 2px 8px rgba(0,0,0,0.5);
                border-left: 4px solid #a29bfe;
            }
        }
        
        .feature-title {
            font-size: 1.2rem;
            font-weight:  600;
            color: #1a1a2e;
            margin-bottom: 0.5rem;
        }
        
        @media (prefers-color-scheme: dark) {
            .feature-title {
                color: #ffffff;
            }
        }
        
        .feature-text {
            font-size: 1rem;
            color: #2d3436;
            line-height: 1.6;
        }
        
        @media (prefers-color-scheme: dark) {
            .feature-text {
                color: #b0b0b0;
            }
        }
        
        .section-header {
            font-size: 1.8rem;
            font-weight:  600;
            color: #1a1a2e;
            margin:  1.5rem 0 1rem 0;
            text-align: center;
        }
        
        @media (prefers-color-scheme: dark) {
            .section-header {
                color: #ffffff;
            }
        }
        
        /* Make selectbox text darker in light mode, lighter in dark mode */
        . stSelectbox label {
            color: #1a1a2e ! important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }
        
        @media (prefers-color-scheme: dark) {
            .stSelectbox label {
                color:  #ffffff !important;
            }
        }
        
        /* Style info boxes */
        .stAlert {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #1a1a2e !important;
            border-left: 4px solid #00b894 !important;
        }
        
        @media (prefers-color-scheme: dark) {
            .stAlert {
                background-color: rgba(30, 30, 46, 0.9) !important;
                color: #e0e0e0 !important;
                border-left: 4px solid #00b894 !important;
            }
        }
        
        . footer-text {
            text-align: center;
            color: #2d3436;
            padding: 1.5rem;
            font-size: 0.95rem;
        }
        
        @media (prefers-color-scheme: dark) {
            .footer-text {
                color: #b0b0b0;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='title'>🧠 Last Minute Prep</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Cram Smart, Not Hard 🚀</div>", unsafe_allow_html=True)

# App Description
st.markdown("""
    <div class='description'>
        Exams tomorrow? No worries! Turn PDFs into cheat sheets, quiz your notes, 
        or speed-watch YouTube lectures. AI does the heavy lifting—you just ace the test 💯
    </div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("<div class='section-header'>✨ What Can You Do?</div>", unsafe_allow_html=True)

col1, col2 = st. columns(2)

with col1:
    st.markdown("""
        <div class='feature-box'>
            <div class='feature-title'>📝 Study Materials</div>
            <div class='feature-text'>
                Upload PDFs or enter topics → Get cheat sheets, quizzes, mnemonics & exam questions.  
                Plus chat with your PDFs!  🚀
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st. markdown("""
        <div class='feature-box'>
            <div class='feature-title'>🎥 Video Notes</div>
            <div class='feature-text'>
                Paste a YouTube link → Get smart summaries or detailed notes. 
                Skip the 2-hour lecture! ⚡
            </div>
        </div>
    """, unsafe_allow_html=True)

# Tool Selection
st.markdown("---")
st.markdown("<div class='section-header'>🚀 Pick Your Tool</div>", unsafe_allow_html=True)

task = st.selectbox(
    "What do you need right now?",
    [
        "Study Materials (PDFs & Topics)",
        "YouTube Summarizer"
    ],
    help="Choose wisely, young padawan 🧙‍♂️"
)

# Display info about selected tool
tool_descriptions = {
    "Study Materials (PDFs & Topics)": "📝 Create cheat sheets, quizzes, mnemonics, important questions, or chat with PDFs",
    "YouTube Summarizer":  "🎥 Get summaries and notes from YouTube videos"
}

st.info(tool_descriptions[task])
st.markdown("---")

# Route to appropriate app
if task == "Study Materials (PDFs & Topics)":
    cheatsheet_app.run_app()

elif task == "YouTube Summarizer":  
    yt_summary_app.run_app()

# Footer
st.markdown("---")
st.markdown(
    "<div class='footer-text'>"
    "Made with 🔥 by students, for students | Powered by LangChain & Groq"
    "</div>",
    unsafe_allow_html=True)
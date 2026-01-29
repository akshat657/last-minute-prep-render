import streamlit as st
from typing import List, Optional, Dict
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO
from reportlab.lib. pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import markdown2
import re
import time

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import RateLimitError  # For proper error handling

# ========= LangChain Compatibility =========
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except: 
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.chains. summarize import load_summarize_chain
except:
    from langchain.chains import load_summarize_chain

try:
    from langchain.chains.question_answering import load_qa_chain
except:
    from langchain. chains import load_qa_chain

try:
    from langchain. prompts import PromptTemplate
except:
    from langchain.prompts. prompt import PromptTemplate
# ==========================================

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"

# ✅ OPTIMIZED LIMITS
MAX_PAGES_PER_PDF = 50
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500
TOP_K_CHUNKS = 4

# Content thresholds for different strategies
SMALL_PDF_THRESHOLD = 15000
MEDIUM_PDF_THRESHOLD = 35000
LARGE_PDF_THRESHOLD = 50000

# Cheatsheet-specific limits (higher than others)
CHEATSHEET_CONTENT_LIMIT = 25000
CHEATSHEET_MAPREDUCE_CHUNK = 10000

SUBJECT_CATEGORIES = {
    "Mathematics": "Math formulas & theorems",
    "Physics": "Physics laws & numericals",
    "Chemistry": "Reactions & concepts",
    "English": "Key ideas & analysis",
    "History": "Dates & events",
    "Biology": "Processes & terms",
    "Computer Science": "Algorithms & complexity",
    "Other": "General revision",
}


def safe_llm_call(model, prompt, operation_name="Processing"):
    """
    Wrapper for LLM calls with proper error handling
    """
    try:
        response = model. invoke(prompt)
        return response. content. strip()
    except RateLimitError as e:
        # Parse the wait time from error message
        error_msg = str(e)
        wait_time_match = re.search(r'try again in ([\d]+[mh][\d]+\.[\d]+s)', error_msg)
        
        if wait_time_match: 
            wait_time = wait_time_match.group(1)
            st.error(f"""
            ⏰ **Rate Limit Reached**
            
            You've used up your daily token limit for Groq API.
            
            **Please try again in: {wait_time}**
            
            💡 **Tips to avoid limits:**
            - Use "Quick Mode" for large PDFs
            - Reduce number of pages/questions
            - The limit resets every 24 hours
            
            🔗 Need more?  [Upgrade to Dev Tier](https://console.groq.com/settings/billing)
            """)
        else:
            st.error(f"""
            ⏰ **Rate Limit Reached**
            
            You've used up your daily token limit (100,000 tokens per day).
            
            **Please try again later** (usually resets in a few hours)
            
            💡 **Tips:**
            - Use "Quick Mode" for faster processing
            - Reduce number of pages or questions
            - Wait for the limit to reset
            
            🔗 [Upgrade for more tokens](https://console.groq.com/settings/billing)
            """)
        
        return None
    except Exception as e:
        st.error(f"❌ {operation_name} failed: {str(e)}")
        return None


def extract_pdf_text(pdf_files: List) -> str:
    """Extract text from PDF files with page limit"""
    output = []
    total_pages_read = 0
    
    for upload in pdf_files:
        reader = PdfReader(upload)
        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, MAX_PAGES_PER_PDF)
        
        st.info(f"📄 {upload.name}:  Reading {pages_to_read} of {total_pages} pages")
        
        for page_num, page in enumerate(reader.pages[: pages_to_read]):
            text = page.extract_text()
            if text:
                output.append(text)
            total_pages_read += 1
    
    st.success(f"✅ Total pages processed: {total_pages_read}")
    return "\n\n".join(output)


def get_text_chunks(text):
    """Split text into optimized chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Create and save vector store"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device':  'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Faiss_index")
    return vector_store


def get_conversational_chain():
    """Enhanced Q&A chain"""
    prompt_template = """
You are a helpful study assistant. Answer the question based on the provided context. 

Instructions:
- Use ONLY the information from the context below
- Be specific and detailed
- If the answer is not in the context, clearly state:  "I cannot find this information in the provided PDF."
- Quote relevant parts when possible

Context:
{context}

Question: 
{question}

Detailed Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    model = ChatGroq(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=1024
    )

    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )
    return chain


def process_content_mapreduce(subject: str, content: str, feature_type: str, **kwargs) -> str:
    """MapReduce strategy for large PDFs with error handling"""
    
    if feature_type == "cheatsheet":
        chunk_size = CHEATSHEET_MAPREDUCE_CHUNK
    else:
        chunk_size = 8000
        
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    
    st.info(f"📚 Processing {len(chunks)} sections with MapReduce...")
    
    model = ChatGroq(model=LLM_MODEL, temperature=0.2)
    
    # MAP Phase
    chunk_summaries = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        progress_text.text(f"⚙️ Processing section {i+1} of {len(chunks)}...")
        
        if feature_type == "cheatsheet": 
            map_prompt = f"""
            Extract EVERYTHING important from this section for {subject}. 
            Focus on: {kwargs.get('subtopics', 'all concepts')}
            
            IMPORTANT: 
            - Extract ALL key concepts, formulas, definitions, facts, and important points
            - Be COMPREHENSIVE - don't skip anything important
            - Format as DENSE bullet points with sub-bullets
            - Include specific numbers, dates, formulas, and examples
            - This will be used for exam preparation
            
            Text:
            {chunk}
            
            Output as many detailed bullet points as possible: 
            """
        elif feature_type == "quiz":
            map_prompt = f"""
            Extract testable concepts and facts for {subject} that can be turned into quiz questions.
            
            Text:
            {chunk}
            """
        elif feature_type == "mnemonics":
            map_prompt = f"""
            Extract difficult concepts and terms from {subject} that need memory aids.
            
            Text:
            {chunk}
            """
        else:  # important questions
            map_prompt = f"""
            Extract important topics and concepts from {subject} that could be exam questions.
            
            Text:
            {chunk}
            """
        
        result = safe_llm_call(model, map_prompt, f"Processing section {i+1}")
        if result is None: 
            progress_bar.empty()
            progress_text.empty()
            return ""
        
        chunk_summaries. append(result)
        progress_bar.progress((i + 1) / len(chunks))
        time.sleep(0.1)
    
    progress_bar.empty()
    progress_text.empty()
    
    # REDUCE Phase
    combined = "\n\n".join(chunk_summaries)
    
    st.success("✅ All sections processed!  Creating final output...")
    
    if feature_type == "cheatsheet":
        reduce_prompt = f"""
        Create a COMPREHENSIVE and DENSE {kwargs.get('num_pages', 2)}-page cheat sheet for {subject}.
        Focus on: {kwargs.get('subtopics', 'all major concepts')}
        
        CRITICAL REQUIREMENTS:
        1. **Maximum Content Density**: Pack as much information as possible
        2. **Comprehensive Coverage**:  Include ALL topics from the extracted points below
        3. **Format for Space Efficiency**:
           - Use short headings (## Topic)
           - Dense bullet points (•) with sub-bullets (◦)
           - Abbreviate where possible (e.g., vs., i.e., etc.)
           - Use symbols (→, ≈, ∴, ∵, ∈, ∀)
           - Put formulas inline:  E=mc², F=ma
        4. **Content Style**:
           - Keywords in **bold**
           - Short definitions (5-10 words max)
           - List key points in compact format
           - Include specific numbers, dates, values
        5. **Structure**:
           - Multiple topics per section
           - 3-4 levels of hierarchy
           - No long paragraphs - only bullet lists
        
        Extracted Key Points (USE ALL OF THESE):
        {combined[: 30000]}
        
        Generate a PACKED cheat sheet with maximum information per page: 
        """
    elif feature_type == "quiz":
        reduce_prompt = f"""
        Generate {kwargs.get('num_questions', 10)} multiple-choice questions for {subject}.
        
        Format EXACTLY as: 
        Q1. [Question]
        A) [Option]
        B) [Option]
        C) [Option]
        D) [Option]
        Answer: A
        
        Based on these concepts: 
        {combined[:18000]}
        """
    elif feature_type == "mnemonics":
        reduce_prompt = f"""
        Create fun memory tricks, mnemonics, and stories for {subject}.
        Make them easy to remember and engaging.
        
        Based on these concepts:
        {combined[:15000]}
        """
    else:  # important questions
        reduce_prompt = f"""
        List {kwargs.get('num_questions', 10)} important exam questions for {subject}.
        Include short answer, long answer, and application-based questions.
        
        Based on these topics:
        {combined[:18000]}
        """
    
    final_result = safe_llm_call(model, reduce_prompt, "Creating final output")
    return final_result if final_result else ""


def process_content_hybrid(subject: str, content: str, feature_type: str, **kwargs) -> str:
    """Hybrid strategy for medium PDFs"""
    content_length = len(content)
    
    st.info(f"📊 Using enhanced sampling from {content_length: ,} characters...")
    
    if feature_type == "cheatsheet":
        first_part = content[:int(content_length * 0.4)]
        middle_start = int(content_length * 0.25)
        middle_end = int(content_length * 0.75)
        middle_part = content[middle_start:middle_end]
        last_part = content[int(content_length * 0.55):]
        
        sampled_content = (
            first_part[: 8000] +
            "\n\n[...  middle content ... ]\n\n" +
            middle_part[: 6000] +
            "\n\n[... later content ...]\n\n" +
            last_part[:6000]
        )
    else:
        first_part = content[:int(content_length * 0.4)]
        middle_start = int(content_length * 0.3)
        middle_end = int(content_length * 0.7)
        middle_part = content[middle_start:middle_end]
        last_part = content[int(content_length * 0.6):]
        
        sampled_content = (
            first_part[:5000] +
            "\n\n[... middle section ...]\n\n" +
            middle_part[:3000] +
            "\n\n[... later section ...]\n\n" +
            last_part[:4000]
        )
    
    st.info(f"📄 Using {len(sampled_content):,} characters from key sections")
    
    return process_content_direct(subject, sampled_content, feature_type, **kwargs)


def process_content_direct(subject: str, content: str, feature_type: str, **kwargs) -> str:
    """Direct processing for small PDFs with error handling"""
    model = ChatGroq(model=LLM_MODEL, temperature=0.2)
    
    if feature_type == "cheatsheet": 
        prompt = f"""
        Create a COMPREHENSIVE and DENSE {kwargs.get('num_pages', 2)}-page cheat sheet for {subject}.
        Focus on: {kwargs.get('subtopics', 'all major concepts')}
        
        CRITICAL REQUIREMENTS FOR CHEAT SHEET:
        
        1. **Maximum Information Density**:
           - Pack MAXIMUM content per page
           - Use compact formatting
           - Multiple topics per section
           - Dense bullet points with sub-bullets
        
        2. **Formatting for Space Efficiency**:
           - Short headings:  ## Topic Name
           - Bullet format: • Main point
             ◦ Sub-point (indented)
             ◦ Another sub-point
           - Use abbreviations:  vs, eg, ie, etc
           - Mathematical symbols: →, ≈, ∴, ±, ≤, ≥
           - Inline formulas: F=ma, E=mc²
        
        3. **Content Style**:
           - **Bold** for key terms
           - *Italic* for emphasis
           - Short definitions (max 10 words)
           - Include specific: 
             • Numbers and values
             • Dates and years
             • Formulas and equations
             • Key examples (1-2 words)
        
        4. **Structure** (Use ALL topics from content):
           ```
           ## Topic 1
           • Key point 1
             ◦ Detail A
             ◦ Detail B
           • Key point 2 with **formula**:  x = y+z
           
           ## Topic 2
           • Concept A → leads to B
           • **Important**: Definition here
             ◦ Example:  brief
           ```
        
        5. **Coverage**:  Include EVERY major topic from the content below
        
        Content (USE ALL OF THIS):
        {content[: CHEATSHEET_CONTENT_LIMIT]}
        
        Generate a PACKED, DENSE cheat sheet with maximum information: 
        """
    elif feature_type == "quiz":
        prompt = f"""
        Generate {kwargs.get('num_questions', 10)} multiple-choice questions for {subject}.
        
        Format EXACTLY as:
        Q1. [Question]
        A) [Option]
        B) [Option]
        C) [Option]
        D) [Option]
        Answer: A
        
        Make questions challenging but fair. 
        
        Content: 
        {content[:15000]}
        """
    elif feature_type == "mnemonics":
        prompt = f"""
        Create memory tricks, mnemonics, or short stories for {subject}.
        Make them fun, relatable, and easy to recall during exams.
        
        Content:
        {content[:15000]}
        """
    else:   # important questions
        prompt = f"""
        List {kwargs. get('num_questions', 10)} potential exam questions for {subject}.
        Include short answer, long answer, and application-based questions.
        
        Content:
        {content[:15000]}
        """
    
    result = safe_llm_call(model, prompt, f"Generating {feature_type}")
    return result if result else ""


def generate_content_with_strategy(subject:  str, content: str, feature_type: str, strategy: str, **kwargs) -> str:
    """Generate content with pre-selected strategy and error handling"""
    result = ""
    
    if strategy == "direct":
        st.success("📄 Small PDF - Processing directly (fastest)")
        result = process_content_direct(subject, content, feature_type, **kwargs)
    
    elif strategy == "hybrid": 
        st.info("📊 Medium PDF - Using smart sampling strategy")
        result = process_content_hybrid(subject, content, feature_type, **kwargs)
    
    elif strategy == "quick":
        st.info("⚡ Quick Mode:  Using first sections for fast processing")
        quick_content = content[:SMALL_PDF_THRESHOLD]
        result = process_content_direct(subject, quick_content, feature_type, **kwargs)
    
    elif strategy == "complete":
        st.info("🔄 Complete Mode: Processing entire PDF with MapReduce")
        result = process_content_mapreduce(subject, content, feature_type, **kwargs)
    
    else:
        result = process_content_direct(subject, content, feature_type, **kwargs)
    
    if not result:
        st.warning("⚠️ Operation cancelled due to rate limit.  Please try again later.")
    
    return result if result else ""


def parse_quiz(quiz_text: str) -> List[Dict]: 
    """Parse quiz text into structured format"""
    questions = []
    current_question = {}
    
    lines = quiz_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if re.match(r'^Q\d+\.  ', line):
            if current_question:
                questions. append(current_question)
            current_question = {
                'question': re.sub(r'^Q\d+\.\s*', '', line),
                'options': {},
                'answer': ''
            }
        elif re.match(r'^[A-D]\)', line):
            option_letter = line[0]
            option_text = line[3: ].strip()
            if current_question: 
                current_question['options'][option_letter] = option_text
        elif line.startswith('Answer: ') or line.startswith('**Answer: '):
            answer_text = re.sub(r'\*\*Answer:\s*|\*\*|Answer:\s*', '', line).strip()
            answer_match = re.search(r'[A-D]', answer_text)
            if answer_match and current_question:
                current_question['answer'] = answer_match.group(0)
    
    if current_question and current_question. get('question'):
        questions.append(current_question)
    
    return questions


def user_input_smart(user_question):
    """Smart PDF Q&A with relevant chunk retrieval and error handling"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        db = FAISS.load_local(
            "Faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = db.similarity_search(user_question, k=TOP_K_CHUNKS)
        
        with st.expander("🔍 Relevant Content Found (Click to view)"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Section {i+1}:**")
                st.text(doc.page_content[: 300] + "...")
                st.markdown("---")
        
        total_chars = sum(len(doc.page_content) for doc in docs)
        st.info(f"📊 Analyzing {total_chars:,} characters from {len(docs)} relevant sections")
        
        chain = get_conversational_chain()
        
        try:
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            st.markdown("### 💡 Answer:")
            st.write(response["output_text"])
            
            if "cannot find" in response["output_text"]. lower():
                st.warning("⚠️ Answer might not be in the PDF")
            else:
                st.success("✅ Answer found in PDF content")
                
        except RateLimitError as e:
            error_msg = str(e)
            wait_time_match = re.search(r'try again in ([\d]+[mh][\d]+\. [\d]+s)', error_msg)
            
            if wait_time_match: 
                wait_time = wait_time_match.group(1)
                st.error(f"""
                ⏰ **Rate Limit Reached**
                
                Please try again in:  **{wait_time}**
                
                💡 The Groq API has a daily limit of 100,000 tokens. 
                """)
            else:
                st.error("⏰ **Rate Limit Reached** - Please try again later (usually resets in a few hours)")

    except FileNotFoundError:
        st. warning("⚠️ Please upload and process PDFs first.")
    except Exception as e: 
        st.error(f"❌ Error:  {str(e)}")


def generate_pdf(markdown_text: str) -> Optional[BytesIO]:
    """Generate PDF from markdown"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    html = markdown2.markdown(markdown_text)
    for line in html.split("\n"):
        clean = re.sub("<[^<]+?>", "", line)
        if clean.strip():
            story. append(Paragraph(clean, styles["Normal"]))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer


def run_app():
    st.markdown("## 📝 Study Assistant - Smart PDF Processing")
    
    with st.expander("⚙️ Current Configuration & Limits"):
        st.write(f"- **Max Pages per PDF:** {MAX_PAGES_PER_PDF} pages")
        st.write(f"- **Small PDF Threshold:** {SMALL_PDF_THRESHOLD: ,} chars (~15 pages)")
        st.write(f"- **Medium PDF Threshold:** {MEDIUM_PDF_THRESHOLD:,} chars (~35 pages)")
        st.write(f"- **Cheatsheet Content Limit:** {CHEATSHEET_CONTENT_LIMIT:,} chars")
        st.write(f"- **Chunk Size:** {CHUNK_SIZE: ,} characters")
        st.write(f"- **Chunks per Question (Q&A):** {TOP_K_CHUNKS}")
        st.markdown("---")
        st.markdown("""
        **Processing Strategies:**
        - 🟢 **Small PDFs:** Direct processing (fastest)
        - 🟡 **Medium PDFs:** Smart sampling (balanced)
        - 🔴 **Large PDFs:** Choose Quick or Complete mode
        """)

    # Initialize session state
    if 'selected_action' not in st.session_state:
        st.session_state.selected_action = None
    if 'quiz_mode' not in st.session_state:
        st.session_state.quiz_mode = None
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False

    # Input method selection
    input_mode = st.radio("Choose input method:", ["📄 Upload PDF", "💭 Enter Topic"], horizontal=True)

    pdf_docs = None
    topic = None
    subject = None

    if input_mode == "📄 Upload PDF":
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
            help=f"Maximum {MAX_PAGES_PER_PDF} pages per PDF will be processed"
        )
        subject = "General Study Material"
    else: 
        topic = st.text_area("Enter your topic", placeholder="e.g., Photosynthesis, Newton's Laws, etc.")
        subject = st. selectbox("📚 Select Subject", SUBJECT_CATEGORIES. keys())

    # Common options
    st.markdown("### ⚙️ Options")
    subtopics = st.text_input("Specific subtopics (optional)", placeholder="e.g., Definitions, Formulas, Examples")

    # Action buttons
    st.markdown("### 🚀 What do you want to do?")
    
    if input_mode == "📄 Upload PDF" and pdf_docs: 
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📋 Cheat Sheet", use_container_width=True):
                st.session_state.selected_action = "cheatsheet"
                st.session_state.quiz_mode = None
        with col2:
            if st. button("🎯 Quiz", use_container_width=True):
                st.session_state. selected_action = "quiz"
                st.session_state.quiz_mode = None
        with col3:
            if st.button("🧠 Mnemonics", use_container_width=True):
                st.session_state.selected_action = "mnemonics"
                st. session_state.quiz_mode = None
        with col4:
            if st.button("❓ Important Qs", use_container_width=True):
                st.session_state.selected_action = "questions"
                st.session_state.quiz_mode = None
        with col5:
            if st.button("💬 Ask PDF", use_container_width=True):
                st.session_state.selected_action = "pdf_qa"
                st.session_state.quiz_mode = None
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋 Cheat Sheet", use_container_width=True):
                st. session_state.selected_action = "cheatsheet"
                st.session_state.quiz_mode = None
        with col2:
            if st.button("🎯 Quiz", use_container_width=True):
                st.session_state.selected_action = "quiz"
                st. session_state.quiz_mode = None
        with col3:
            if st.button("🧠 Mnemonics", use_container_width=True):
                st.session_state.selected_action = "mnemonics"
                st.session_state. quiz_mode = None
        with col4:
            if st. button("❓ Important Qs", use_container_width=True):
                st.session_state.selected_action = "questions"
                st.session_state.quiz_mode = None

    # Validate inputs
    has_valid_input = False
    content = ""
    
    if input_mode == "📄 Upload PDF" and pdf_docs:
        has_valid_input = True
        with st.spinner("📖 Reading PDF..."):
            content = extract_pdf_text(pdf_docs)
        if not subject:
            subject = "General Study Material"
    elif input_mode == "💭 Enter Topic" and topic:
        has_valid_input = True
        content = f"Create materials on:  {topic}"
        if not subject:
            subject = "Other"

    # Show specific options based on selected action
    if st.session_state.selected_action and has_valid_input:
        st.markdown("---")
        
        # Determine strategy based on content size
        content_length = len(content)
        
        if content_length <= SMALL_PDF_THRESHOLD:
            selected_strategy = "direct"
            strategy_info = "📄 Small PDF - Will process directly (fastest)"
        elif content_length <= MEDIUM_PDF_THRESHOLD:
            selected_strategy = "hybrid"
            strategy_info = "📊 Medium PDF - Will use smart sampling"
        else: 
            selected_strategy = None
            strategy_info = f"📚 Large PDF ({content_length:,} characters) - Choose your strategy below"
        
        if selected_strategy:
            st. info(strategy_info)
        else:
            st.warning(strategy_info)
        
        if st.session_state.selected_action == "cheatsheet":
            st.markdown("### 📋 Cheat Sheet Options")
            
            num_pages = st.slider("Number of pages:", 1, 5, 2, key="cheat_pages")
            
            if not selected_strategy:
                strategy_choice = st.radio(
                    "⚡ Choose processing strategy:",
                    [
                        "Quick Mode (First sections only - ~10 seconds)",
                        "Complete Mode (Entire PDF with MapReduce - ~1-2 minutes)"
                    ],
                    help="Quick mode processes beginning sections.  Complete mode analyzes entire PDF."
                )
                
                if "Quick Mode" in strategy_choice:
                    selected_strategy = "quick"
                else:
                    selected_strategy = "complete"
            
            if st.button("✨ Generate Cheat Sheet", use_container_width=True, type="primary"):
                with st.spinner("🔄 Creating your cheat sheet..."):
                    subtopics_text = subtopics if subtopics else "all major concepts"
                    result = generate_content_with_strategy(
                        subject,
                        content,
                        "cheatsheet",
                        selected_strategy,
                        num_pages=num_pages,
                        subtopics=subtopics_text
                    )
                    
                    if result:
                        st.markdown("### 📋 Your Cheat Sheet")
                        st.markdown(result)
                        
                        col1, col2 = st. columns(2)
                        with col1:
                            st. download_button("📄 Download Markdown", result, file_name="cheatsheet.md")
                        with col2:
                            pdf_file = generate_pdf(result)
                            if pdf_file:
                                st.download_button("📕 Download PDF", pdf_file, file_name="cheatsheet. pdf")
        
        elif st.session_state. selected_action == "quiz": 
            st.markdown("### 🎯 Quiz Options")
            
            num_questions = st.slider("Number of questions:", 5, 20, 10, key="quiz_qs")
            
            if not selected_strategy:
                strategy_choice = st.radio(
                    "⚡ Choose processing strategy:",
                    [
                        "Quick Mode (First sections only - ~10 seconds)",
                        "Complete Mode (Entire PDF with MapReduce - ~1-2 minutes)"
                    ],
                    key="quiz_strategy"
                )
                
                if "Quick Mode" in strategy_choice:
                    selected_strategy = "quick"
                else:
                    selected_strategy = "complete"
            
            if st.session_state.quiz_mode is None:
                st.info("Choose how you want to use the quiz:")
                col1, col2 = st. columns(2)
                
                with col1:
                    if st.button("🎮 Take Interactive Test", use_container_width=True, type="primary"):
                        st.session_state.quiz_mode = "interactive"
                        st.rerun()
                
                with col2:
                    if st. button("📄 Download Quiz", use_container_width=True):
                        st.session_state.quiz_mode = "download"
                        st.rerun()
            
            elif st.session_state.quiz_mode == "interactive":
                if st.session_state.quiz_data is None:
                    with st.spinner("🔄 Creating your quiz..."):
                        quiz_text = generate_content_with_strategy(
                            subject,
                            content,
                            "quiz",
                            selected_strategy,
                            num_questions=num_questions
                        )
                        
                        if quiz_text:
                            parsed_quiz = parse_quiz(quiz_text)
                            
                            if not parsed_quiz:
                                st.error("❌ Failed to parse quiz. Please try again.")
                                if st.button("🔄 Retry"):
                                    st.session_state.quiz_mode = None
                                    st.rerun()
                            else:
                                st.session_state.quiz_data = parsed_quiz
                                st.rerun()
                        else: 
                            st.session_state.quiz_mode = None
                
                if st.session_state.quiz_data:
                    st.markdown("### 🎮 Interactive Quiz")
                    st.info("Select your answers and click Submit to see your score!")
                    
                    for idx, q_data in enumerate(st.session_state.quiz_data):
                        st.markdown(f"#### Question {idx + 1}")
                        st. write(q_data['question'])
                        
                        options_list = [f"{k}) {v}" for k, v in q_data['options'].items()]
                        
                        if not st.session_state.quiz_submitted:
                            selected = st.radio(
                                f"Select your answer:",
                                options_list,
                                key=f"q_{idx}",
                                index=None
                            )
                            if selected:
                                st.session_state.user_answers[idx] = selected[0]
                        else:
                            user_ans = st.session_state.user_answers.get(idx, "")
                            correct_ans = q_data['answer']
                            
                            for opt in options_list:
                                opt_letter = opt[0]
                                if opt_letter == correct_ans:
                                    st.success(f"✅ {opt} (Correct Answer)")
                                elif opt_letter == user_ans and user_ans != correct_ans: 
                                    st.error(f"❌ {opt} (Your Answer)")
                                else: 
                                    st.write(opt)
                        
                        st.markdown("---")
                    
                    if not st.session_state.quiz_submitted:
                        if st.button("📊 Submit Quiz", use_container_width=True, type="primary"):
                            st.session_state.quiz_submitted = True
                            st.rerun()
                    else: 
                        correct_count = 0
                        total = len(st.session_state.quiz_data)
                        
                        for idx, q_data in enumerate(st.session_state.quiz_data):
                            if st.session_state.user_answers.get(idx) == q_data['answer']: 
                                correct_count += 1
                        
                        score_percentage = (correct_count / total) * 100
                        
                        st.markdown("### 🎯 Your Results")
                        
                        if score_percentage >= 80:
                            st.success(f"🎉 Excellent! You scored {correct_count}/{total} ({score_percentage:.1f}%)")
                        elif score_percentage >= 60:
                            st.info(f"👍 Good job! You scored {correct_count}/{total} ({score_percentage:.1f}%)")
                        else:
                            st.warning(f"📚 Keep practicing! You scored {correct_count}/{total} ({score_percentage:. 1f}%)")
                        
                        if st.button("🔄 Take Another Quiz", use_container_width=True):
                            st.session_state.quiz_data = None
                            st.session_state.user_answers = {}
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_mode = None
                            st.rerun()
            
            elif st.session_state. quiz_mode == "download":
                with st.spinner("🔄 Creating quiz... "):
                    result = generate_content_with_strategy(
                        subject,
                        content,
                        "quiz",
                        selected_strategy,
                        num_questions=num_questions
                    )
                    
                    if result: 
                        st.markdown("### 🎯 Your Quiz")
                        st.markdown(result)
                        
                        st.download_button("📄 Download Quiz", result, file_name="quiz.md")
                    
                    if st.button("🔄 Back to Options"):
                        st.session_state.quiz_mode = None
                        st.rerun()
        
        elif st.session_state.selected_action == "mnemonics":
            st.markdown("### 🧠 Memory Tricks & Mnemonics")
            st.info("Creating fun memory aids to help you remember key concepts!")
            
            if not selected_strategy:
                strategy_choice = st.radio(
                    "⚡ Choose processing strategy:",
                    [
                        "Quick Mode (First sections only - ~10 seconds)",
                        "Complete Mode (Entire PDF with MapReduce - ~1-2 minutes)"
                    ],
                    key="mnemonics_strategy"
                )
                
                if "Quick Mode" in strategy_choice:
                    selected_strategy = "quick"
                else:
                    selected_strategy = "complete"
            
            if st.button("✨ Generate Mnemonics", use_container_width=True, type="primary"):
                with st.spinner("🔄 Creating memory tricks..."):
                    result = generate_content_with_strategy(
                        subject,
                        content,
                        "mnemonics",
                        selected_strategy
                    )
                    
                    if result:
                        st.markdown("### 🧠 Your Memory Tricks")
                        st. markdown(result)
                        
                        st.download_button("📄 Download Mnemonics", result, file_name="mnemonics.md")
        
        elif st.session_state.selected_action == "questions": 
            st.markdown("### ❓ Important Exam Questions")
            
            num_questions = st.slider("Number of questions:", 5, 20, 10, key="imp_qs")
            
            if not selected_strategy:
                strategy_choice = st.radio(
                    "⚡ Choose processing strategy:",
                    [
                        "Quick Mode (First sections only - ~10 seconds)",
                        "Complete Mode (Entire PDF with MapReduce - ~1-2 minutes)"
                    ],
                    key="questions_strategy"
                )
                
                if "Quick Mode" in strategy_choice:
                    selected_strategy = "quick"
                else:
                    selected_strategy = "complete"
            
            if st.button("✨ Generate Questions", use_container_width=True, type="primary"):
                with st.spinner("🔄 Generating important questions..."):
                    result = generate_content_with_strategy(
                        subject,
                        content,
                        "questions",
                        selected_strategy,
                        num_questions=num_questions
                    )
                    
                    if result: 
                        st.markdown("### ❓ Potential Exam Questions")
                        st. markdown(result)
                        
                        st.download_button("📄 Download Questions", result, file_name="important_questions.md")
        
        elif st.session_state.selected_action == "pdf_qa": 
            st.markdown("### 💬 Ask Your PDF")
            st.info("💡 Smart Q&A:  Only relevant content is sent to AI, not the entire PDF!")
            
            if 'pdf_processed' not in st.session_state:
                with st.spinner("🔄 Processing PDFs for Q&A..."):
                    text_chunks = get_text_chunks(content)
                    st.success(f"✅ Created {len(text_chunks)} searchable chunks")
                    get_vector_store(text_chunks)
                    st.session_state['pdf_processed'] = True
                    st.session_state['pdf_text'] = content
                    st.success("✅ PDF processed!  Ask your questions below.")
            
            user_question = st.text_input("🗣️ Ask a question about your PDF:")
            
            if user_question:
                user_input_smart(user_question)
    
    elif st.session_state.selected_action and not has_valid_input:
        st.warning("⚠️ Please upload a PDF or enter a topic first!")


if __name__ == "__main__":
    run_app()
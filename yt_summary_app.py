import os
import re
import streamlit as st
from typing import List
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    RequestBlocked,   # <- new
)
from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig  # <- new


from langchain. text_splitter import RecursiveCharacterTextSplitter
from langchain. prompts import PromptTemplate
from langchain. chains. summarize import load_summarize_chain
from langchain_core. documents. base import Document
from langchain_groq import ChatGroq
from groq import RateLimitError

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Missing GROQ_API_KEY environment variable")
def _get_ytt_api():
    """
    Initialize YouTubeTranscriptApi with proxy support if credentials are present.
    Priority:
     1) Webshare rotating residential proxies via WEBSHARE_USERNAME / WEBSHARE_PASSWORD
     2) Generic HTTP/HTTPS proxies via HTTP_PROXY / HTTPS_PROXY
     3) Default direct mode (no proxy)
    """
    # Webshare rotating residential proxy (recommended if you purchased residential plan)
    ws_user = os.getenv("WEBSHARE_USERNAME")
    ws_pass = os.getenv("WEBSHARE_PASSWORD")
    ws_locations = os.getenv("WEBSHARE_LOCATIONS")  # optional comma-separated, e.g. "us,de"

    if ws_user and ws_pass:
        locs = None
        if ws_locations:
            locs = [loc.strip() for loc in ws_locations.split(",") if loc.strip()]
        try:
            if locs:
                proxy_cfg = WebshareProxyConfig(
                    proxy_username=ws_user,
                    proxy_password=ws_pass,
                    filter_ip_locations=locs
                )
            else:
                proxy_cfg = WebshareProxyConfig(
                    proxy_username=ws_user,
                    proxy_password=ws_pass
                )
            return YouTubeTranscriptApi(proxy_config=proxy_cfg)
        except Exception as e:
            # Fall back to non-proxy if initializing Webshare config fails,
            # but log a message for debugging.
            st.warning(f"Could not initialize Webshare proxy: {e}. Falling back to default fetch (may be blocked).")
            return YouTubeTranscriptApi()

    # Generic proxies (HTTP / HTTPS)
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if http_proxy or https_proxy:
        try:
            proxy_cfg = GenericProxyConfig(http_url=http_proxy, https_url=https_proxy)
            return YouTubeTranscriptApi(proxy_config=proxy_cfg)
        except Exception as e:
            st.warning(f"Could not initialize generic proxy: {e}. Falling back to default fetch.")
            return YouTubeTranscriptApi()

    # Default, direct (no proxy)
    return YouTubeTranscriptApi()


map_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "🔍 Here's a transcript section (~500 words):\n\n"
        "{text}\n\n"
        "Summarize the key points as bullet points (≤250 words)."
    ),
)

combine_prompt = PromptTemplate(
    input_variables=["text"],
    template="📚 Now combine all these bullet lists into one coherent summary:\n\n{text}",
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.25,
    max_retries=2,
)

summarizer = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    verbose=False,
)


def extract_video_id(url_or_id: str) -> str | None:
    s = url_or_id.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    parsed = urlparse(s)
    h = (parsed.netloc or "").lower()

    if "youtu.be" in h:
        return parsed.path.lstrip("/").split("?")[0]

    if "youtube.com" in h:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0]

        if parsed.path.startswith("/live/"):
            return parsed.path.split("/live/")[-1]. split("?")[0]

        m = re.search(r"/(? :embed|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if m:
            return m.group(1)

    return None


def extract_transcript(url: str, languages: List[str] = ["en", "hi"]) -> str:
    vid = extract_video_id(url)
    if not vid:
        st.error("❌ Could not parse a valid YouTube video ID.")
        return ""

    ytt_api = _get_ytt_api()

    try:
        fetched = ytt_api.fetch(
            video_id=vid,
            languages=languages,
        )
    except TranscriptsDisabled:
        st.error("🎙️ Captions disabled or unavailable for this video.")
        return ""
    except NoTranscriptFound:
        st.error("❓ Transcript not found for requested languages.")
        return ""
    except RequestBlocked as e:
        # Specific error when YouTube blocks the IP (common on cloud hosts)
        st.error(
            "🚫 YouTube blocked transcript access from this server (RequestBlocked).\n\n"
            "Possible fixes:\n"
            "  • If you have Webshare rotating residential proxies, set WEBSHARE_USERNAME and WEBSHARE_PASSWORD as env vars.\n"
            "  • Or paste the transcript manually (or upload a .txt) — use the manual input option.\n\n"
            "Note: Using proxies may incur cost and YouTube may still block some IPs. Proxies are optional."
        )
        print("YouTube RequestBlocked:", e)
        return ""
    except Exception as e:
        # Generic fallback (includes IpBlocked, network errors, etc.)
        st.error(
            "Unexpected error fetching transcript: "
            f"{e}\n\nThis often means YouTube blocked requests from this server.\n"
            "If you want to use rotating residential proxies, set WEBSHARE_USERNAME and WEBSHARE_PASSWORD in your environment."
        )
        print("Unexpected transcript fetch error:", e)
        return ""

    return " ".join(snippet.text.strip() for snippet in fetched)


def summarize_transcript(transcript: str) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
    )

    docs = [
        Document(page_content=chunk)
        for chunk in splitter. split_text(transcript)
    ]

    try:
        result = summarizer. invoke({"input_documents": docs})
        return result["output_text"]. strip()

    except RateLimitError as e:
        st.error(
            "❌ Rate limit exceeded:  You've used up today's 100,000 token quota for Groq.  "
            "Please try again later."
        )
        print("Groq RateLimitError:", e)
        return ""

    except Exception as e:
        st.error(f"Unexpected error during summarization: {e}")
        print("Unexpected summarization error:", e)
        return ""


def create_notes_from_transcript(transcript: str, note_type: str, num_pages: int) -> str:
    if note_type == "Concise":
        style_instruction = "Create concise, to-the-point notes with only essential information."
    elif note_type == "Detailed":
        style_instruction = "Create detailed, comprehensive notes with explanations and examples."
    else:  # Bullet-point
        style_instruction = "Create notes in bullet-point format with clear hierarchical structure."

    prompt = f"""
Create {num_pages}-page study notes from this video transcript.
Style: {style_instruction}
Include main topics, key concepts, and important points.
Make it easy to review for exams. 

Transcript:
{transcript[: 10000]}
"""

    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.25)
    
    try:
        response = model.invoke(prompt)
        return str(response.content).strip()
    except RateLimitError: 
        st.error("❌ Rate limit exceeded. Try again later.")
        return ""
    except Exception as e:
        st.error(f"Error creating notes: {e}")
        return ""


def run_app():
    st.markdown("## 🎥 YouTube Summarizer & Notes")

    youtube_input = st.text_input("YouTube URL or Video ID:", placeholder="https://youtube.com/watch?v=...")

    # Mode selection
    mode = st.radio("What do you want to create?", ["📝 Quick Summary", "📓 Study Notes"], horizontal=True)

    if mode == "📓 Study Notes":
        col1, col2 = st. columns(2)
        
        with col1:
            note_type = st.selectbox("Note style:", ["Concise", "Detailed", "Bullet-point"])
        
        with col2:
            num_pages = st.slider("Number of pages:", 1, 5, 2)

    if youtube_input and st.button("🚀 Generate", use_container_width=True):
        transcript = extract_transcript(youtube_input)
        if not transcript: 
            return

        if mode == "📝 Quick Summary": 
            with st.spinner("🔎 Summarizing video..."):
                summary = summarize_transcript(transcript)

            if summary:
                st.markdown("## 📝 Summary")
                st.write(summary)
                st.download_button("📄 Download Summary", summary, file_name="video_summary.md")
        
        else:  # Study Notes
            with st.spinner(f"📓 Creating {note_type. lower()} notes..."):
                notes = create_notes_from_transcript(transcript, note_type, num_pages)

            if notes:
                st.markdown(f"## 📓 Your {note_type} Notes")
                st.markdown(notes)
                st.download_button("📄 Download Notes", notes, file_name="video_notes.md")


if __name__ == "__main__": 
    run_app()
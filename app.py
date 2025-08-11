import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from gtts import gTTS
import tempfile
import os

# ✅ Configure Streamlit page
st.set_page_config(page_title="AI Notes Summarizer", layout="centered")

# ✅ Load summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ✅ Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ✅ Summarize text in chunks
def summarize_text(text):
    if len(text.strip()) == 0:
        return "No text found in the document."
    
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary += "• " + result[0]['summary_text'].strip() + "\n\n"
    return summary

# ✅ Generate audio from summary
def generate_audio(summary_text):
    tts = gTTS(summary_text)
    temp_path = os.path.join(tempfile.gettempdir(), "summary_audio.mp3")
    tts.save(temp_path)
    return temp_path

# ✅ UI Layout
st.title("🧠 AI Notes Summarizer with Audio")
st.write("Upload a PDF file, get a summarized version of your notes, and even listen to it!")

uploaded_file = st.file_uploader("📄 Upload Your PDF Notes", type=["pdf"])

if uploaded_file:
    if "raw_text" not in st.session_state:
        with st.spinner("📖 Extracting text from PDF..."):
            st.session_state.raw_text = extract_text_from_pdf(uploaded_file)

    st.subheader("📘 Extracted Text Preview")
    st.text_area("Original Notes", st.session_state.raw_text[:1000] + "...", height=200)

    if st.button("📝 Generate Summary"):
        with st.spinner("🧠 Summarizing..."):
            st.session_state.summary = summarize_text(st.session_state.raw_text)

    if "summary" in st.session_state:
        st.success("✅ Summary generated!")
        st.subheader("📋 Summary Output")
        st.text_area("Summary", st.session_state.summary, height=250)

        if st.button("🔊 Generate Audio Summary"):
            with st.spinner("🎙️ Creating audio..."):
                audio_file = generate_audio(st.session_state.summary)

            st.subheader("🔉 Audio Summary")
            st.audio(audio_file)

            with open(audio_file, "rb") as f:
                st.download_button("⬇️ Download Audio", f, file_name="summary_audio.mp3", mime="audio/mpeg")

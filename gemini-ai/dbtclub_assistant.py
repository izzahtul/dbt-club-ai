from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# model yang digunakan di yt,pdf, dan chatbot
text_model = genai.GenerativeModel("gemini-pro")
# model yang digunakan di image
image_model = genai.GenerativeModel("gemini-pro-vision")
chat = text_model.start_chat(history=[])

def get_gemini_text_response(input_text):
    response = text_model.generate_content(input_text)
    return response.text


def get_gemini_image_response(image):
    response = image_model.generate_content([image])
    return response.text

def get_chat_response(question):
    response = chat.send_message(question, stream=True)
    return response

# extract video ID from a YouTube URL
def extract_video_id(youtube_url):
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
    if video_id_match:
        return video_id_match.group(1)
    return None

# extract transcript 
def extract_transcript_details(youtube_video_url):
    try:
        video_id = extract_video_id(youtube_video_url)
        if not video_id:
            st.error("Invalid YouTube URL.")
            return None
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except NoTranscriptFound:
        try:
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=['id'])
            transcript = " ".join([i["text"] for i in transcript_text])
            return transcript
        except NoTranscriptFound:
            st.error("No suitable transcript found.")
            return None
    except Exception as e:
        st.error(f"Error extracting transcript: {e}")
        return None

# function summary dari transcript
def generate_gemini_content(transcript_text, prompt):
    response = text_model.generate_content(prompt + transcript_text)
    return response.text

# function qna based context yt
def answer_question(transcript_text, question, qa_prompt):
    prompt = qa_prompt.format(transcript=transcript_text, question=question)
    response = text_model.generate_content(prompt)
    return response.text

# function extract pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# function split text pdf 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# function untuk create dan save - FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# function untuk QnA PDF
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the
    provided context, just say, "The answer is not available in your PDF", and don't provide a wrong answer. You can also summarize the PDF files if the user asks.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# function user input untuk pdf
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Streamlit app configuration
st.set_page_config(page_title="DBTClub Assistant")
st.header("DBTClub Assistant")

# tab masing-masing fungsi AI
tab1, tab2, tab3, tab4 = st.tabs(["Text & Image Input", "Chat Assistant", "YouTube Summarizer & Q&A", "PDF Q&A"])

with tab1:
    st.subheader("Text & Image Input")
    input_text = st.text_input("Input Prompt:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    image = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    if st.button("Tell me about the image"):
        if input_text:
            response = get_gemini_text_response(input_text)
            st.write(response)
        if image:
            image_response = get_gemini_image_response(image)
            st.write(image_response)

with tab2:
    st.subheader("Chat Assistant")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    chat_input = st.text_input("Input:")
    if st.button("Ask DBT Assistant") and chat_input:
        chat_response = get_chat_response(chat_input)
        st.session_state['chat_history'].append(("You", chat_input))
        st.subheader("The Response is")
        for chunk in chat_response:
            st.write(chunk.text)
            st.session_state['chat_history'].append(("Bot", chunk.text))
    st.subheader("Chat History")
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")

with tab3:
    st.subheader("YouTube Summarizer & Q&A")
    youtube_link = st.text_input("Enter YouTube Video Link:")

    if youtube_link:
        video_id = extract_video_id(youtube_link)
        if video_id:
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
            if 'transcript_text' not in st.session_state:
                transcript_text = extract_transcript_details(youtube_link)
                if transcript_text:
                    st.session_state.transcript_text = transcript_text
                else:
                    st.session_state.transcript_text = None
        else:
            st.error("Invalid YouTube URL.")

    if 'transcript_text' in st.session_state and st.session_state.transcript_text:
        summary_prompt = """You are a YouTube video summarizer. You will be taking the transcript text and summarizing the entire video
        and providing the important summary in points within 500 words.
        Please provide the summary of the text given here: """
        
        if 'summary' not in st.session_state:
            summary = generate_gemini_content(st.session_state.transcript_text, summary_prompt)
            st.session_state.summary = summary
        st.markdown("## Detailed Notes:")
        st.write(st.session_state.summary)

        qa_prompt = """You are an assistant for YouTube video content. You will answer questions based on the provided video transcript.
        Transcript: {transcript}
        Question: {question}
        Answer: """

        question = st.text_input("Ask a question about the video:")
        if st.button("Get Answer"):
            if question:
                answer = answer_question(st.session_state.transcript_text, question, qa_prompt)
                st.markdown("## Answer to your question:")
                st.write(answer)
    else:
        if youtube_link and not st.session_state.get('transcript_text'):
            st.error("Unable to extract transcript.")
        elif not youtube_link:
            st.error("Please enter a valid YouTube link.")

with tab4:
    st.subheader("PDF Q&A")
    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])
    
    if st.button("Submit"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete")
        else:
            st.error("Please upload at least one PDF file.")

    user_question = st.text_input("Ask a question from your PDF files:")
    
    if user_question:
        user_input(user_question)

from PyPDF2 import PdfReader
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

client = Groq(api_key = os.getenv("groq_api_key"))

async def get_pdf_text(documents):
    logging.info("Started Pdf data extraction...")
    text = ""
    for pdf in documents:
        contents = await pdf.read()
        pdf_reader = PdfReader(BytesIO(contents))
        for page in pdf_reader.pages:
            text += page.extract_text()
                
    return text

def get_chunks(text):
    logging.info("Started Chunking...")
    tex_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = tex_splitter.split_text(text)
    return chunks
    
def generate_embeddings(chunks):
    logging.info("Started embedding generation and saving into FAISS...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
    vector_store = FAISS.from_texts(chunks, embedding_model)
    
    # 2 approach

    # model = SentenceTransformer("all-MiniLM-L6-V2")
    # vector_store = model.encode(chunks).astype("float32")
    # docs = [Document(page_content=chunk) for chunk in chunks]
    # vector_store = FAISS.from_texts(vector_store, docs)
    
    return vector_store
    
def get_response_from_llm(retrieved_texts,Query):
    logging.info("Call llm to get answer from llm based on provided knowledge base document")
    prompt = f"""
        You are an assistant. Use ONLY the context below to answer the question,fo not anything extra info. if we don't have answer in context simply say i dont know.

        --- BEGIN CONTEXT ---
        {retrieved_texts}
        --- END CONTEXT ---

        Question: {Query}

        Answer:
    """
        
    response = client.chat.completions.create(
        model = "openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Answer based on context only."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        temperature=0.7,
        top_p=1,
        stream=False,
        stop=None
    )

    logging.info("response:"+response.choices[0].message.content)
    answer = response.choices[0].message.content
        
    return answer
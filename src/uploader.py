from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse 
import logging
from typing import List
from src.helper import get_pdf_text, get_chunks, generate_embeddings, get_response_from_llm
from dotenv import load_dotenv
load_dotenv()
import os

logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1",
    responses={404: {"description":"Not Found"}}
)

@router.post("/upload_document",tags=["upload"])
async def upload_doc(documents: List[UploadFile] = File(...), Query: str = Form(...)):
    try:
        logging.info("Starting")
        print("Starting.....")
        text = await get_pdf_text(documents)
        if not text:
            return ""
        
        text_chunks = get_chunks(text)
        
        vector_store = generate_embeddings(text_chunks)
        
        docs = vector_store.similarity_search(Query, k=5)
        
        retrieved_texts = ""
        for page in docs:
            logging.info(f"page_content: {page.page_content}")
            retrieved_texts += page.page_content+"\n"
            
        answer = get_response_from_llm(retrieved_texts,Query)
        
        return JSONResponse(content={"answer": answer}, status_code=200)
    except Exception as e:
        logging.info(f"Exception: {e}")
        return JSONResponse(content={"error": f"Exception Occurred: {e}"}, status_code=500)
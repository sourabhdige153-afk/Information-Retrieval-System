from fastapi import FastAPI
import uvicorn
from src import uploader
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="My information retrieval app",
    version='1.0.0',
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    root_path="/myapp"
)


app.add_middleware(CORSMiddleware, allow_origins=["*"],allow_credentials=True, allow_methods=["*"],allow_headers=["*"])


app.include_router(uploader.router)

@app.get("/hello_world")
def read_root():
    return {"message: Hello world"}

if __name__ == "__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000)
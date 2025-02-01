from chatbot import chat
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

app = FastAPI(
    title="Chatbot API",
    description="This is a chatbot API",
    version="0.1",
    docs_url="/",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ask")
def ask(question:str):
    response = chat(question)
    return JSONResponse(content={"response": response})
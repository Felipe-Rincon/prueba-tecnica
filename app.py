from fastapi import FastAPI, Request
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
import subprocess

app = FastAPI()

@app.middleware("http")
async def add_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.update({
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'self'"
    })
    return response

@app.get("/")
async def home():
    subprocess.Popen(["streamlit", "run", "app_streamlit.py", "--server.port=8501"])
    return {"message": "Streamlit se está ejecutando en segundo plano..."}
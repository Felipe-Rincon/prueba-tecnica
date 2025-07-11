# from fastapi import FastAPI, Request
# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
# import subprocess
# #streamlit run app.py --server.port=$PORT
# app = FastAPI()

# @app.middleware("http")
# async def add_headers(request: Request, call_next):
#     response = await call_next(request)
#     response.headers.update({
#         "X-Frame-Options": "DENY",
#         "X-Content-Type-Options": "nosniff",
#         "Content-Security-Policy": "default-src 'self'"
#     })
#     return response

# @app.get("/")
# async def home():
#     subprocess.Popen(["streamlit", "run", "app_streamlit.py", "--server.port=8501"])
#     return {"message": "Streamlit se está ejecutando en segundo plano..."}
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import os
import subprocess

app = FastAPI()

STREAMLIT_URL = "http://localhost:8501"  # Streamlit interno

# Inicia Streamlit en segundo plano al arrancar
subprocess.Popen([
    "streamlit", "run", "app_streamlit.py",
    "--server.port=8501",
    "--server.headless=true",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false"
])

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

@app.api_route("/{path:path}", methods=["GET", "POST", "WEBSOCKET"])
async def proxy_streamlit(request: Request, path: str):
    async with httpx.AsyncClient(base_url=STREAMLIT_URL) as client:
        # Reenvía la petición a Streamlit
        streamlit_response = await client.request(
            request.method,
            f"/{path}",
            params=request.query_params,
            content=await request.body(),
            headers=dict(request.headers)
        )
        return StreamingResponse(
            content=streamlit_response.iter_bytes(),
            status_code=streamlit_response.status_code,
            headers=dict(streamlit_response.headers)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
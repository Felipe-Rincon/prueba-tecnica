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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import httpx
import os
import subprocess
import uvicorn

app = FastAPI()

# Configuración de Streamlit
STREAMLIT_PORT = "8501"
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

# Iniciar Streamlit en segundo plano
subprocess.Popen([
    "streamlit", "run", "app_streamlit.py",
    f"--server.port={STREAMLIT_PORT}",
    "--server.headless=true",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false"
])

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.update({
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'self'"
    })
    return response

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "WEBSOCKET"])
async def proxy_to_streamlit(request: Request, path: str):
    async with httpx.AsyncClient(base_url=STREAMLIT_URL) as client:
        try:
            response = await client.request(
                request.method,
                f"/{path}",
                headers=dict(request.headers),
                params=request.query_params,
                content=await request.body()
            )
            return HTMLResponse(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        except httpx.ConnectError:
            return HTMLResponse("Streamlit backend not available", status_code=503)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
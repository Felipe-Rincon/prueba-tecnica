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
from fastapi.responses import RedirectResponse
import httpx
import uvicorn

app = FastAPI()

STREAMLIT_URL = "http://localhost:8501"  # URL interna de Streamlit

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # Cabeceras de seguridad
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy_to_streamlit(request: Request, path: str):
    # Redirige todas las peticiones a Streamlit
    async with httpx.AsyncClient(base_url=STREAMLIT_URL) as client:
        response = await client.request(
            request.method,
            f"/{path}",
            params=request.query_params,
            content=await request.body(),
        )
        return RedirectResponse(url=STREAMLIT_URL)

if __name__ == "__main__":
    import subprocess
    # Inicia Streamlit en segundo plano
    subprocess.Popen(["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.headless=true"])
    # Inicia FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8080)
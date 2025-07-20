#!/bin/bash
# Instalar NGINX si no existe
if ! command -v nginx &> /dev/null; then
    apt-get update && apt-get install -y nginx
fi

# Iniciar NGINX en segundo plano
nginx -c /app/nginx.conf -g "daemon off;" &

# Iniciar Streamlit
streamlit run app_streamlit.py --server.port=8501 --server.headless=true --server.enableCORS=false
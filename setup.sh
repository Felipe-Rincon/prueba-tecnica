#!/bin/bash
# Instala NGINX y envsubst (para reemplazar $PORT)
apt-get update && apt-get install -y nginx gettext-base

# Genera la configuración final de NGINX
envsubst '$PORT' < nginx.conf > /etc/nginx/nginx.conf

# Inicia NGINX en segundo plano


# Inicia Streamlit en el puerto interno 8501
nginx & streamlit run app.py --server.port=8501 --server.headless=true
#!/bin/bash
# Instala herramientas para reemplazar variables
apt-get update && apt-get install -y gettext-base

# Reemplaza $PORT en nginx.conf
envsubst '$PORT' < /app/nginx.conf > /tmp/nginx.conf
mv /tmp/nginx.conf /etc/nginx/nginx.conf

# Inicia NGINX
nginx -g "daemon off;" &

# Inicia Streamlit en otro puerto
streamlit run app_streamlit.py --server.port=8501 --server.headless=true
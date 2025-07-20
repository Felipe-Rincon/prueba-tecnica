#!/bin/bash

# Instalar NGINX (forzando la instalación aunque Railway ya lo tenga)
apt-get update && apt-get install -y --reinstall nginx

# Reemplazar $PORT en tiempo real (crucial para Railway)
envsubst '$PORT' < /app/nginx.conf > /etc/nginx/nginx.conf

# Iniciar NGINX en primer plano
nginx -g "daemon off;" &

# Iniciar Streamlit en SEGUNDO PLANO (con &)
streamlit run /app/app_streamlit.py \
    --server.port=8501 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true &
    
# Mantener el contenedor en ejecución
tail -f /dev/null
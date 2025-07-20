#!/bin/bash

# --- 1. Instalar NGINX y herramientas esenciales ---
apt-get update && apt-get install -y nginx gettext-base apt-utils

# --- 2. Crear directorio de configuración si no existe ---
mkdir -p /etc/nginx/conf.d/

# --- 3. Reemplazar $PORT usando envsubst ---
# (Asegúrate de que nginx.conf esté en /app)
envsubst '$PORT' < /app/nginx.conf > /tmp/nginx.conf
mv /tmp/nginx.conf /etc/nginx/nginx.conf  # Mueve el archivo generado

# --- 4. Iniciar NGINX ---
nginx -g "daemon off;" &

# --- 5. Iniciar Streamlit ---
streamlit run app.py --server.port=8501 --server.headless=true
# # Usa una imagen base con Python y Ubuntu
# FROM python:3.9-slim

# # Instala NGINX y dependencias del sistema
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     nginx \
#     gettext-base && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# # Copia los archivos al contenedor
# COPY . /app
# WORKDIR /app

# # Instala dependencias de Python
# RUN pip install --no-cache-dir -r requirements.txt

# # Configura NGINX (reemplaza variables como $PORT)
# RUN envsubst '$PORT' < nginx.conf > /etc/nginx/nginx.conf

# # Expone el puerto de NGINX (Railway inyectará el puerto real)
# EXPOSE 80

# # Comando de inicio (NGINX + Streamlit)
# CMD service nginx start && \
#     streamlit run app.py --server.port=8501 --server.headless=true

# Usa una imagen base con Python y Debian (slim pero con soporte para nginx-extras)
FROM python:3.9-slim

# Instala NGINX con módulos esenciales (incluyendo headers-more) + dependencias
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    nginx-extras \ 
    gettext-base \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos al contenedor
COPY . /app
WORKDIR /app

# Instala dependencias de Python (asegúrate de incluir todas en requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Configura NGINX (reemplaza $PORT y asegura permisos)
RUN envsubst '$PORT' < nginx.conf > /etc/nginx/nginx.conf && \
    chmod 644 /etc/nginx/nginx.conf 

# Expone el puerto 80 (Railway manejará el puerto externo)
EXPOSE 80

# Comando de inicio (NGINX + Streamlit)
CMD ["sh", "-c", "service nginx start && streamlit run app.py --server.port=8501 --server.headless=true --server.enableCORS=false"]
#!/bin/bash

# Instalar NGINX si no está presente
if ! command -v nginx &> /dev/null; then
    apt-get update && apt-get install -y nginx
fi

# Reemplazar $PORT en la configuración
envsubst '$PORT' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# Iniciar NGINX
nginx -g "daemon on;"

# Esperar un momento para que NGINX se inicie
sleep 2
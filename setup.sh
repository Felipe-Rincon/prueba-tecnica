#!/bin/bash

# Instalar NGINX si no está presente
if ! command -v nginx &> /dev/null; then
    apt-get update && apt-get install -y nginx
fi

# Copiar configuración de NGINX
cp nginx.conf /etc/nginx/nginx.conf

# Iniciar NGINX en segundo plano
nginx -g "daemon off;" &
#!/bin/bash
# ============================================================================
# Módulo: 05-docker-compose.sh
# Propósito: Generar y configurar docker-compose.yml para VibeVoice
# Versión: 1.0.0
# Descripción: Crea archivo docker-compose completo con todos los servicios
# ============================================================================

set -euo pipefail

# Cargar dependencias
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configuracion/stack-vibe.conf"
source "${SCRIPT_DIR}/../librerias/registrador.sh"
source "${SCRIPT_DIR}/../librerias/ayudante.sh"
source "${SCRIPT_DIR}/../librerias/validador.sh"

# ============================================================================
# FUNCIÓN: configurar_docker_compose
# ============================================================================
configurar_docker_compose() {
    registrar_inicio_modulo "Configuración de Docker Compose"
    
    # Crear directorio de configuración
    crear_directorio "${VIBE_DIR_CONFIG}"
    
    # Generar archivo docker-compose.yml
    generar_docker_compose_file
    
    # Generar archivo .env
    generar_env_file
    
    # Validar archivo docker-compose
    validar_docker_compose_file
    
    registrar_fin_modulo "Configuración de Docker Compose"
    return 0
}

# ============================================================================
# FUNCIÓN: generar_docker_compose_file
# ============================================================================
generar_docker_compose_file() {
    local compose_file="${VIBE_DIR_CONFIG}/docker-compose.yml"
    
    registrar_info "Generando archivo docker-compose.yml..."
    
    cat > "${compose_file}" <<'EOF'
version: '3.8'

networks:
  vibevoice-network:
    external: true

volumes:
  postgres-data:
  redis-data:
  kafka-data:
  zookeeper-data:

services:
  # ============================================================================
  # PostgreSQL - Base de datos principal
  # ============================================================================
  postgres:
    image: postgres:${POSTGRES_VERSION:-15-alpine}
    container_name: vibevoice-postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_MAX_CONNECTIONS: ${POSTGRES_MAX_CONNECTIONS:-200}
      POSTGRES_SHARED_BUFFERS: ${POSTGRES_SHARED_BUFFERS:-256MB}
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - vibevoice-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ============================================================================
  # Redis - Caché y cola de mensajes
  # ============================================================================
  redis:
    image: redis:${REDIS_VERSION:-7-alpine}
    container_name: vibevoice-redis
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory ${REDIS_MAX_MEMORY:-2gb}
      --maxmemory-policy ${REDIS_EVICTION_POLICY:-allkeys-lru}
      --appendonly yes
      --appendfsync everysec
    ports:
      - "${REDIS_PORT}:6379"
    volumes:
      - redis-data:/data
    networks:
      - vibevoice-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # ============================================================================
  # Zookeeper - Coordinación para Kafka
  # ============================================================================
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    container_name: vibevoice-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: ${KAFKA_ZOOKEEPER_PORT:-2181}
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "${KAFKA_ZOOKEEPER_PORT:-2181}:${KAFKA_ZOOKEEPER_PORT:-2181}"
    volumes:
      - zookeeper-data:/var/lib/zookeeper/data
    networks:
      - vibevoice-network
    restart: unless-stopped

  # ============================================================================
  # Kafka - Sistema de mensajería
  # ============================================================================
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    container_name: vibevoice-kafka
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:${KAFKA_ZOOKEEPER_PORT:-2181}
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:${KAFKA_PORT}
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_HEAP_OPTS: ${KAFKA_HEAP_OPTS:--Xmx1G -Xms1G}
    ports:
      - "${KAFKA_PORT}:${KAFKA_PORT}"
    volumes:
      - kafka-data:/var/lib/kafka/data
    networks:
      - vibevoice-network
    restart: unless-stopped

  # ============================================================================
  # VibeVoice API - Servicio principal
  # ============================================================================
  vibevoice-api:
    image: vibevoice:latest
    container_name: vibevoice-api
    depends_on:
      - postgres
      - redis
      - kafka
    environment:
      # Base de datos
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      
      # Redis
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
      
      # Kafka
      KAFKA_BOOTSTRAP_SERVERS: kafka:${KAFKA_PORT}
      
      # Configuración de la API
      API_HOST: 0.0.0.0
      API_PORT: ${API_PORT:-8000}
      API_WORKERS: ${API_WORKERS:-4}
      API_TIMEOUT: ${API_TIMEOUT:-300}
      
      # Seguridad
      SECRET_KEY: ${SECRET_KEY}
      JWT_ALGORITHM: ${JWT_ALGORITHM:-HS256}
      JWT_EXPIRATION: ${JWT_EXPIRATION:-3600}
      
      # CORS
      CORS_ORIGINS: ${CORS_ORIGINS}
      
      # Logs
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
    ports:
      - "${API_PORT:-8000}:${API_PORT:-8000}"
    volumes:
      - ${VIBE_DIR_BASE}/app:/app
      - ${VIBE_DIR_LOGS}:/app/logs
    networks:
      - vibevoice-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:${API_PORT:-8000}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF

    registrar_exito "Archivo docker-compose.yml generado: ${compose_file}"
    registrar_auditoria "Docker Compose configurado en: ${compose_file}"
    
    return 0
}

# ============================================================================
# FUNCIÓN: generar_env_file
# ============================================================================
generar_env_file() {
    local env_file="${VIBE_DIR_CONFIG}/.env"
    
    registrar_info "Generando archivo .env..."
    
    cat > "${env_file}" <<EOF
# ============================================================================
# Configuración de Entorno para VibeVoice
# ADVERTENCIA: Este archivo contiene credenciales sensibles
# NO compartir ni subir a control de versiones
# ============================================================================

# PostgreSQL
POSTGRES_VERSION=${VIBE_POSTGRES_VERSION}
POSTGRES_DB=${VIBE_POSTGRES_DB}
POSTGRES_USER=${VIBE_POSTGRES_USER}
POSTGRES_PASSWORD=${VIBE_POSTGRES_PASSWORD}
POSTGRES_PORT=${VIBE_POSTGRES_PORT}
POSTGRES_MAX_CONNECTIONS=${VIBE_POSTGRES_MAX_CONNECTIONS}
POSTGRES_SHARED_BUFFERS=${VIBE_POSTGRES_SHARED_BUFFERS}

# Redis
REDIS_VERSION=${VIBE_REDIS_VERSION}
REDIS_PORT=${VIBE_REDIS_PORT}
REDIS_PASSWORD=${VIBE_REDIS_PASSWORD}
REDIS_MAX_MEMORY=${VIBE_REDIS_MAX_MEMORY}
REDIS_EVICTION_POLICY=${VIBE_REDIS_EVICTION_POLICY}

# Kafka
KAFKA_PORT=${VIBE_KAFKA_PORT}
KAFKA_ZOOKEEPER_PORT=${VIBE_KAFKA_ZOOKEEPER_PORT}
KAFKA_HEAP_OPTS=${VIBE_KAFKA_HEAP_OPTS}

# API
API_PORT=${VIBE_API_PORT}
API_WORKERS=${VIBE_API_WORKERS}
API_TIMEOUT=${VIBE_API_TIMEOUT}

# Seguridad
SECRET_KEY=${VIBE_SECRET_KEY}
JWT_ALGORITHM=${VIBE_JWT_ALGORITHM}
JWT_EXPIRATION=${VIBE_JWT_EXPIRATION}

# CORS
CORS_ORIGINS=${VIBE_CORS_ORIGINS}

# Logs
LOG_LEVEL=${VIBE_LOG_LEVEL}

# Directorios
VIBE_DIR_BASE=${VIBE_DIR_BASE}
VIBE_DIR_LOGS=${VIBE_DIR_LOGS}
EOF

    chmod 600 "${env_file}"
    
    registrar_exito "Archivo .env generado: ${env_file}"
    registrar_advertencia "IMPORTANTE: El archivo .env contiene credenciales. Protéjalo adecuadamente."
    
    # Crear también .env.example (sin credenciales sensibles)
    local env_example="${VIBE_DIR_CONFIG}/.env.example"
    registrar_info "Generando archivo .env.example..."
    
    cat > "${env_example}" <<'ENVEXAMPLE'
# ============================================================================
# Ejemplo de Configuración de Entorno para VibeVoice
# IMPORTANTE: Copiar a .env y ajustar valores según entorno
# ============================================================================

# PostgreSQL
POSTGRES_VERSION=15-alpine
POSTGRES_DB=vibevoice_db
POSTGRES_USER=vibe_admin
POSTGRES_PASSWORD=CHANGE_ME_POSTGRES_PASSWORD
POSTGRES_PORT=5432
POSTGRES_MAX_CONNECTIONS=200
POSTGRES_SHARED_BUFFERS=256MB

# Redis
REDIS_VERSION=7-alpine
REDIS_PORT=6379
REDIS_PASSWORD=CHANGE_ME_REDIS_PASSWORD
REDIS_MAX_MEMORY=2gb
REDIS_EVICTION_POLICY=allkeys-lru

# Kafka
KAFKA_PORT=9092
KAFKA_ZOOKEEPER_PORT=2181
KAFKA_HEAP_OPTS=-Xmx1G -Xms1G

# API
API_PORT=8000
API_WORKERS=1
API_TIMEOUT=300

# Seguridad
SECRET_KEY=CHANGE_ME_SECRET_KEY_FOR_PRODUCTION
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Logs
LOG_LEVEL=INFO

# Directorios
VIBE_DIR_BASE=/opt/vibevoice
VIBE_DIR_LOGS=/opt/vibevoice/logs

# Performance (para modo desarrollo local ligero)
TRANSCRIBE_MODEL=whisper-tiny
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
UVICORN_WORKERS=1
UVICORN_LIMIT_CONCURRENCY=1
ENVEXAMPLE

    chmod 644 "${env_example}"
    registrar_exito "Archivo .env.example generado: ${env_example}"
    
    return 0
}

# ============================================================================
# FUNCIÓN: validar_docker_compose_file
# ============================================================================
validar_docker_compose_file() {
    local compose_file="${VIBE_DIR_CONFIG}/docker-compose.yml"
    
    registrar_info "Validando archivo docker-compose.yml..."
    
    cd "${VIBE_DIR_CONFIG}"
    
    if docker-compose config --quiet 2>/dev/null || docker compose config --quiet 2>/dev/null; then
        registrar_exito "Archivo docker-compose.yml validado correctamente"
        return 0
    else
        registrar_error "Error en la validación de docker-compose.yml"
        return 1
    fi
}

# ============================================================================
# EJECUCIÓN
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    configurar_docker_compose
fi

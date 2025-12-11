#!/bin/bash
# ============================================================================
# Módulo: 04-servicios-datos.sh
# Propósito: Configurar servicios de datos (PostgreSQL, Redis, Kafka)
# Versión: 1.0.0
# Descripción: Despliega y configura bases de datos y servicios de mensajería
# ============================================================================

set -euo pipefail

# Cargar dependencias
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configuracion/stack-vibe.conf"
source "${SCRIPT_DIR}/../librerias/registrador.sh"
source "${SCRIPT_DIR}/../librerias/ayudante.sh"
source "${SCRIPT_DIR}/../librerias/validador.sh"

# ============================================================================
# FUNCIÓN: configurar_servicios_datos
# ============================================================================
configurar_servicios_datos() {
    registrar_inicio_modulo "Configuración de Servicios de Datos"
    
    # Crear directorios de datos
    crear_directorios_datos

    # Asegurar la existencia de la red Docker requerida (por si el módulo 03 no la creó)
    registrar_info "Verificando red Docker ${VIBE_DOCKER_NETWORK}..."
    if command -v docker &>/dev/null; then
        if ! docker network ls | grep -q "${VIBE_DOCKER_NETWORK}"; then
            ejecutar_comando \
                "docker network create --driver bridge --subnet ${VIBE_DOCKER_SUBNET} ${VIBE_DOCKER_NETWORK}" \
                "Creando red Docker ${VIBE_DOCKER_NETWORK}"
        else
            registrar_info "Red Docker ${VIBE_DOCKER_NETWORK} ya existe"
        fi
    else
        registrar_advertencia "Docker no está disponible; los contenedores no se podrán crear hasta que Docker esté instalado"
    fi
    
    # Configurar PostgreSQL
    configurar_postgresql
    
    # Configurar Redis
    configurar_redis
    
    # Configurar Kafka
    configurar_kafka
    
    registrar_fin_modulo "Configuración de Servicios de Datos"
    return 0
}

# ============================================================================
# FUNCIÓN: crear_directorios_datos
# ============================================================================
crear_directorios_datos() {
    registrar_info "Creando directorios de datos..."
    # Asegurar que el usuario de servicio exista antes de cambiar propietarios
    if ! crear_usuario_sistema "${VIBE_SERVICE_USER}" "${VIBE_DIR_BASE}"; then
        registrar_advertencia "No se pudo crear el usuario ${VIBE_SERVICE_USER}; se usarán permisos root para crear directorios"
    fi
    
    local directorios=(
        "${VIBE_DIR_DATOS}/postgresql"
        "${VIBE_DIR_DATOS}/redis"
        "${VIBE_DIR_DATOS}/kafka"
        "${VIBE_DIR_DATOS}/zookeeper"
    )
    
    for dir in "${directorios[@]}"; do
        crear_directorio "${dir}" "${VIBE_SERVICE_USER:-root}" "755"
    done
    
    registrar_exito "Directorios de datos creados"
}

# ============================================================================
# FUNCIÓN: configurar_postgresql
# ============================================================================
configurar_postgresql() {
    registrar_info "Configurando PostgreSQL..."
    
    # Verificar si el contenedor existe
    if docker ps -a | grep -q "vibevoice-postgres"; then
        local estado=$(docker inspect -f '{{.State.Status}}' vibevoice-postgres 2>/dev/null || echo "unknown")
        
        if [[ "${estado}" == "running" ]]; then
            registrar_info "Contenedor PostgreSQL ya está en ejecución"
            # Verificar si está saludable
            if wait_for_postgres "vibevoice-postgres" 10; then
                registrar_exito "PostgreSQL ya está configurado y funcionando"
                return 0
            else
                registrar_advertencia "PostgreSQL está corriendo pero no responde. Recreando..."
                docker stop vibevoice-postgres 2>/dev/null || true
                docker rm vibevoice-postgres 2>/dev/null || true
            fi
        elif [[ "${estado}" == "exited" ]]; then
            registrar_advertencia "Contenedor PostgreSQL existe pero está detenido. Eliminando y recreando..."
            docker rm vibevoice-postgres 2>/dev/null || true
        else
            registrar_info "Eliminando contenedor PostgreSQL en estado: ${estado}"
            docker stop vibevoice-postgres 2>/dev/null || true
            docker rm vibevoice-postgres 2>/dev/null || true
        fi
    fi
    
    # Crear contenedor PostgreSQL
    registrar_info "Creando contenedor PostgreSQL ${VIBE_POSTGRES_VERSION}..."
    registrar_info "Límites de recursos: Memoria=${POSTGRES_MEM}, CPUs=${POSTGRES_CPUS}"
    
    docker run -d \
        --name vibevoice-postgres \
        --network "${VIBE_DOCKER_NETWORK}" \
        --memory="${POSTGRES_MEM}" \
        --cpus="${POSTGRES_CPUS}" \
        -e POSTGRES_DB="${VIBE_POSTGRES_DB}" \
        -e POSTGRES_USER="${VIBE_POSTGRES_USER}" \
        -e POSTGRES_PASSWORD="${VIBE_POSTGRES_PASSWORD}" \
        -e POSTGRES_MAX_CONNECTIONS="${VIBE_POSTGRES_MAX_CONNECTIONS}" \
        -e POSTGRES_SHARED_BUFFERS="${VIBE_POSTGRES_SHARED_BUFFERS}" \
        -p "${VIBE_POSTGRES_PORT}:5432" \
        -v "${VIBE_DIR_DATOS}/postgresql:/var/lib/postgresql/data" \
        --restart unless-stopped \
        "postgres:${VIBE_POSTGRES_VERSION}" || {
            registrar_error "Error al crear contenedor PostgreSQL"
            return 1
        }
    
    registrar_exito "Contenedor PostgreSQL creado"
    
    # Usar la función wait_for_postgres
    if ! wait_for_postgres "vibevoice-postgres" 60; then
        registrar_error "PostgreSQL no está listo después de 60 segundos"
        return 1
    fi
    
    # Crear base de datos y extensiones
    registrar_info "Configurando base de datos..."
    docker exec vibevoice-postgres psql -U "${VIBE_POSTGRES_USER}" -d "${VIBE_POSTGRES_DB}" -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";" 2>/dev/null || true
    docker exec vibevoice-postgres psql -U "${VIBE_POSTGRES_USER}" -d "${VIBE_POSTGRES_DB}" -c "CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";" 2>/dev/null || true
    
    registrar_exito "PostgreSQL configurado correctamente"
    registrar_auditoria "PostgreSQL desplegado en puerto ${VIBE_POSTGRES_PORT}"
    
    return 0
}

# ============================================================================
# FUNCIÓN: configurar_redis
# ============================================================================
configurar_redis() {
    registrar_info "Configurando Redis..."
    
    # Verificar si el contenedor existe
    if docker ps -a | grep -q "vibevoice-redis"; then
        local estado=$(docker inspect -f '{{.State.Status}}' vibevoice-redis 2>/dev/null || echo "unknown")
        
        if [[ "${estado}" == "running" ]]; then
            registrar_info "Contenedor Redis ya está en ejecución"
            # Verificar si está saludable
            if wait_for_redis "vibevoice-redis" 10; then
                registrar_exito "Redis ya está configurado y funcionando"
                return 0
            else
                registrar_advertencia "Redis está corriendo pero no responde. Recreando..."
                docker stop vibevoice-redis 2>/dev/null || true
                docker rm vibevoice-redis 2>/dev/null || true
            fi
        elif [[ "${estado}" == "exited" ]]; then
            registrar_advertencia "Contenedor Redis existe pero está detenido. Eliminando y recreando..."
            docker rm vibevoice-redis 2>/dev/null || true
        else
            registrar_info "Eliminando contenedor Redis en estado: ${estado}"
            docker stop vibevoice-redis 2>/dev/null || true
            docker rm vibevoice-redis 2>/dev/null || true
        fi
    fi
    
    # Crear contenedor Redis
    registrar_info "Creando contenedor Redis ${VIBE_REDIS_VERSION}..."
    registrar_info "Límites de recursos: Memoria=${REDIS_MEM}, CPUs=${REDIS_CPUS}"
    
    docker run -d \
        --name vibevoice-redis \
        --network "${VIBE_DOCKER_NETWORK}" \
        --memory="${REDIS_MEM}" \
        --cpus="${REDIS_CPUS}" \
        -p "${VIBE_REDIS_PORT}:6379" \
        -v "${VIBE_DIR_DATOS}/redis:/data" \
        --restart unless-stopped \
        "redis:${VIBE_REDIS_VERSION}" \
        redis-server \
        --requirepass "${VIBE_REDIS_PASSWORD}" \
        --maxmemory "${VIBE_REDIS_MAX_MEMORY}" \
        --maxmemory-policy "${VIBE_REDIS_EVICTION_POLICY}" \
        --appendonly yes \
        --appendfsync everysec || {
            registrar_error "Error al crear contenedor Redis"
            return 1
        }
    
    registrar_exito "Contenedor Redis creado"
    
    # Esperar a que Redis esté listo
    registrar_info "Esperando a que Redis esté listo..."
    local intentos=0
    local max_intentos=30
    
    # Usar la función wait_for_redis
    if ! wait_for_redis "vibevoice-redis" 30; then
        registrar_error "Redis no está listo después de 30 segundos"
        return 1
    fi
    
    registrar_exito "Redis configurado correctamente"
    registrar_auditoria "Redis desplegado en puerto ${VIBE_REDIS_PORT}"
    
    return 0
}

# ============================================================================
# FUNCIÓN: configurar_kafka
# ============================================================================
configurar_kafka() {
    registrar_info "Configurando Kafka y Zookeeper..."
    
    # Detener contenedores existentes
    for container in vibevoice-zookeeper vibevoice-kafka; do
        if docker ps -a | grep -q "${container}"; then
            registrar_info "Deteniendo contenedor ${container}..."
            docker stop "${container}" 2>/dev/null || true
            docker rm "${container}" 2>/dev/null || true
        fi
    done
    
    # Crear contenedor Zookeeper
    registrar_info "Creando contenedor Zookeeper..."
    
    docker run -d \
        --name vibevoice-zookeeper \
        --network "${VIBE_DOCKER_NETWORK}" \
        -e ZOOKEEPER_CLIENT_PORT="${VIBE_KAFKA_ZOOKEEPER_PORT}" \
        -e ZOOKEEPER_TICK_TIME=2000 \
        -p "${VIBE_KAFKA_ZOOKEEPER_PORT}:${VIBE_KAFKA_ZOOKEEPER_PORT}" \
        -v "${VIBE_DIR_DATOS}/zookeeper:/var/lib/zookeeper/data" \
        --restart unless-stopped \
        "confluentinc/cp-zookeeper:7.5.0" || {
            registrar_error "Error al crear contenedor Zookeeper"
            return 1
        }
    
    registrar_exito "Contenedor Zookeeper creado"
    
    # Esperar a que Zookeeper esté listo
    sleep 10
    
    # Crear contenedor Kafka
    registrar_info "Creando contenedor Kafka ${VIBE_KAFKA_VERSION}..."
    
    docker run -d \
        --name vibevoice-kafka \
        --network "${VIBE_DOCKER_NETWORK}" \
        -e KAFKA_BROKER_ID=1 \
        -e KAFKA_ZOOKEEPER_CONNECT="vibevoice-zookeeper:${VIBE_KAFKA_ZOOKEEPER_PORT}" \
        -e KAFKA_ADVERTISED_LISTENERS="PLAINTEXT://localhost:${VIBE_KAFKA_PORT}" \
        -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP="PLAINTEXT:PLAINTEXT" \
        -e KAFKA_INTER_BROKER_LISTENER_NAME=PLAINTEXT \
        -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
        -e KAFKA_HEAP_OPTS="${VIBE_KAFKA_HEAP_OPTS}" \
        -e KAFKA_AUTO_CREATE_TOPICS_ENABLE=true \
        -p "${VIBE_KAFKA_PORT}:${VIBE_KAFKA_PORT}" \
        -v "${VIBE_DIR_DATOS}/kafka:/var/lib/kafka/data" \
        --restart unless-stopped \
        "confluentinc/cp-kafka:7.5.0" || {
            registrar_error "Error al crear contenedor Kafka"
            return 1
        }
    
    registrar_exito "Contenedor Kafka creado"
    
    # Esperar a que Kafka esté listo
    registrar_info "Esperando a que Kafka esté listo..."
    sleep 15
    
    # Crear topics de Kafka
    registrar_info "Creando topics de Kafka..."
    IFS=',' read -ra TOPICS <<< "${VIBE_KAFKA_TOPICS}"
    
    for topic in "${TOPICS[@]}"; do
        docker exec vibevoice-kafka kafka-topics \
            --create \
            --if-not-exists \
            --bootstrap-server localhost:${VIBE_KAFKA_PORT} \
            --topic "${topic}" \
            --partitions 3 \
            --replication-factor 1 2>/dev/null || {
                registrar_advertencia "No se pudo crear topic: ${topic}"
            }
        registrar_info "Topic creado: ${topic}"
    done
    
    registrar_exito "Kafka configurado correctamente"
    registrar_auditoria "Kafka desplegado en puerto ${VIBE_KAFKA_PORT}"
    
    return 0
}

# ============================================================================
# EJECUCIÓN
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    configurar_servicios_datos
fi

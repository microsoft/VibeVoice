#!/bin/bash
# ============================================================================
# Módulo: 03-docker.sh
# Propósito: Instalar y configurar Docker y Docker Compose
# Versión: 1.0.0
# Descripción: Instala Docker Engine y Docker Compose para contenedorización
# ============================================================================

set -euo pipefail

# Cargar dependencias
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configuracion/stack-vibe.conf"
source "${SCRIPT_DIR}/../librerias/registrador.sh"
source "${SCRIPT_DIR}/../librerias/ayudante.sh"
source "${SCRIPT_DIR}/../librerias/validador.sh"

# ============================================================================
# FUNCIÓN: instalar_docker
# ============================================================================
instalar_docker() {
    registrar_inicio_modulo "Instalación de Docker"
    
    # Verificar si Docker ya está instalado
    if verificar_comando_existe docker; then
        if validar_docker_version; then
            registrar_info "Docker ya está instalado en versión adecuada"
            if validar_docker_funcionando; then
                registrar_fin_modulo "Instalación de Docker (ya instalado)"
                return 0
            fi
        else
            registrar_advertencia "Docker está instalado pero en versión antigua. Actualizando..."
        fi
    fi
    
    # Actualizar repositorios
    actualizar_apt || registrar_error_fatal "Error al actualizar repositorios"
    
    # Instalar dependencias previas
    registrar_info "Instalando dependencias de Docker..."
    local dependencias=(
        "ca-certificates"
        "curl"
        "gnupg"
        "lsb-release"
        "apt-transport-https"
    )
    
    for paquete in "${dependencias[@]}"; do
        instalar_paquete_apt "${paquete}" || {
            registrar_error "Error al instalar ${paquete}"
            return 1
        }
    done
    
    # Agregar clave GPG oficial de Docker
    registrar_info "Agregando clave GPG de Docker..."
    crear_directorio "/etc/apt/keyrings"
    
    if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
        ejecutar_comando \
            "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg" \
            "Descargando clave GPG de Docker"
        
        chmod a+r /etc/apt/keyrings/docker.gpg
    else
        registrar_info "Clave GPG de Docker ya existe"
    fi
    
    # Configurar repositorio de Docker
    registrar_info "Configurando repositorio de Docker..."
    local docker_repo_file="/etc/apt/sources.list.d/docker.list"
    
    if [[ ! -f "${docker_repo_file}" ]]; then
        echo \
            "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
            $(lsb_release -cs) stable" | tee "${docker_repo_file}" > /dev/null
        
        registrar_exito "Repositorio de Docker configurado"
    else
        registrar_info "Repositorio de Docker ya existe"
    fi
    
    # Actualizar índice de paquetes
    actualizar_apt
    
    # Instalar Docker Engine
    registrar_info "Instalando Docker Engine..."
    local paquetes_docker=(
        "docker-ce"
        "docker-ce-cli"
        "containerd.io"
        "docker-buildx-plugin"
        "docker-compose-plugin"
    )
    
    for paquete in "${paquetes_docker[@]}"; do
        instalar_paquete_apt "${paquete}" || {
            registrar_error "Error al instalar ${paquete}"
            return 1
        }
    done
    
    # Iniciar y habilitar servicio Docker
    registrar_info "Iniciando servicio Docker..."
    ejecutar_comando "systemctl start docker" "Iniciando Docker"
    ejecutar_comando "systemctl enable docker" "Habilitando Docker al inicio"
    
    # Esperar a que Docker esté listo
    esperar_servicio "docker" 30 || {
        registrar_error_fatal "Docker no se inició correctamente"
    }
    
    # Configurar Docker para usuario de servicio
    if [[ -n "${VIBE_SERVICE_USER:-}" ]]; then
        registrar_info "Agregando usuario ${VIBE_SERVICE_USER} al grupo docker..."
        usermod -aG docker "${VIBE_SERVICE_USER}" 2>/dev/null || true
    fi
    
    # Agregar usuario actual al grupo docker si no es root
    if [[ $(whoami) != "root" ]]; then
        usermod -aG docker "$(whoami)"
        registrar_info "Usuario $(whoami) agregado al grupo docker"
    fi
    
    # Configurar daemon de Docker
    registrar_info "Configurando daemon de Docker..."
    local daemon_json="/etc/docker/daemon.json"
    
    if [[ ! -f "${daemon_json}" ]]; then
        cat > "${daemon_json}" <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "live-restore": true,
  "userland-proxy": false
}
EOF
        registrar_exito "Configuración de daemon Docker creada"
        
        # Recargar daemon
        ejecutar_comando "systemctl daemon-reload" "Recargando daemon"
        ejecutar_comando "systemctl restart docker" "Reiniciando Docker"
        esperar_servicio "docker" 30
    fi
    
    # Crear red de Docker para VibeVoice
    registrar_info "Creando red Docker para VibeVoice..."
    if ! docker network ls | grep -q "${VIBE_DOCKER_NETWORK}"; then
        ejecutar_comando \
            "docker network create --driver bridge --subnet ${VIBE_DOCKER_SUBNET} ${VIBE_DOCKER_NETWORK}" \
            "Creando red Docker ${VIBE_DOCKER_NETWORK}"
    else
        registrar_info "Red Docker ${VIBE_DOCKER_NETWORK} ya existe"
    fi
    
    # Verificar instalación
    if validar_docker_funcionando; then
        registrar_exito "Docker instalado y funcionando correctamente"
    else
        registrar_error_fatal "Docker no está funcionando correctamente"
    fi
    
    # Instalar Docker Compose standalone (por compatibilidad)
    instalar_docker_compose_standalone
    
    # Mostrar resumen
    echo ""
    registrar_info "═══════════════════════════════════════════════════════════════"
    registrar_info "RESUMEN DE INSTALACIÓN DE DOCKER"
    registrar_info "═══════════════════════════════════════════════════════════════"
    registrar_info "Versión de Docker: $(docker --version | awk '{print $3}' | sed 's/,//')"
    registrar_info "Versión de Docker Compose: $(docker compose version | awk '{print $4}')"
    registrar_info "Red Docker: ${VIBE_DOCKER_NETWORK}"
    registrar_info "Estado del servicio: $(systemctl is-active docker)"
    registrar_info "═══════════════════════════════════════════════════════════════"
    echo ""
    
    registrar_fin_modulo "Instalación de Docker"
    return 0
}

# ============================================================================
# FUNCIÓN: instalar_docker_compose_standalone
# ============================================================================
instalar_docker_compose_standalone() {
    registrar_info "Instalando Docker Compose standalone..."
    
    # Verificar si ya está instalado
    if verificar_comando_existe docker-compose; then
        if validar_docker_compose_version; then
            registrar_info "Docker Compose standalone ya está instalado"
            return 0
        fi
    fi
    
    # Descargar última versión de Docker Compose
    local compose_version="v2.24.0"
    local compose_url="https://github.com/docker/compose/releases/download/${compose_version}/docker-compose-$(uname -s)-$(uname -m)"
    local compose_dest="/usr/local/bin/docker-compose"
    
    registrar_info "Descargando Docker Compose ${compose_version}..."
    ejecutar_comando \
        "curl -L '${compose_url}' -o '${compose_dest}'" \
        "Descargando Docker Compose"
    
    chmod +x "${compose_dest}"
    
    # Crear enlace simbólico
    if [[ ! -L /usr/bin/docker-compose ]]; then
        ln -sf "${compose_dest}" /usr/bin/docker-compose
    fi
    
    registrar_exito "Docker Compose standalone instalado: $(docker-compose --version)"
    return 0
}

# ============================================================================
# EJECUCIÓN
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    instalar_docker
fi

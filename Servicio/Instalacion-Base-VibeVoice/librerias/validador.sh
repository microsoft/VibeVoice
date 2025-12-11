#!/bin/bash
# ============================================================================
# Archivo: validador.sh
# Propósito: Funciones de validación de requisitos y configuración
# Versión: 1.0.0
# Descripción: Valida requisitos del sistema, versiones y configuraciones
# ============================================================================

# Cargar dependencias
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configuracion/stack-vibe.conf" 2>/dev/null || true
source "${SCRIPT_DIR}/registrador.sh" 2>/dev/null || true
source "${SCRIPT_DIR}/ayudante.sh" 2>/dev/null || true

# ============================================================================
# FUNCIÓN: validar_ubuntu_version
# Propósito: Validar que se esté ejecutando en Ubuntu 22.04 o 24.04
# Retorno: 0 si la versión es válida, 1 si no lo es
# ============================================================================
validar_ubuntu_version() {
    registrar_info "Validando versión de Ubuntu..."
    
    if [[ ! -f /etc/os-release ]]; then
        registrar_error "No se pudo detectar el sistema operativo"
        return 1
    fi
    
    source /etc/os-release
    
    if [[ "${ID}" != "ubuntu" ]]; then
        registrar_error "Este instalador solo funciona en Ubuntu. Detectado: ${ID}"
        return 1
    fi
    
    local version_valida=false
    for version in ${VIBE_UBUNTU_VERSIONS}; do
        if [[ "${VERSION_ID}" == "${version}" ]]; then
            version_valida=true
            break
        fi
    done
    
    if [[ "${version_valida}" == "true" ]]; then
        registrar_exito "Versión de Ubuntu válida: ${VERSION_ID} (${PRETTY_NAME})"
        return 0
    else
        registrar_error "Versión de Ubuntu no soportada: ${VERSION_ID}"
        registrar_error "Versiones soportadas: ${VIBE_UBUNTU_VERSIONS}"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: validar_recursos_sistema
# Propósito: Validar que el sistema cumpla con los requisitos mínimos
# Retorno: 0 si cumple los requisitos, 1 si no los cumple
# ============================================================================
validar_recursos_sistema() {
    registrar_info "Validando recursos del sistema..."
    local errores=0
    
    # Validar RAM
    local ram_gb=$(obtener_memoria_total_gb)
    registrar_info "RAM detectada: ${ram_gb} GB"
    
    if [[ ${ram_gb} -lt ${VIBE_MIN_RAM_GB} ]]; then
        registrar_error "RAM insuficiente: ${ram_gb} GB (mínimo: ${VIBE_MIN_RAM_GB} GB)"
        ((errores++))
    else
        registrar_exito "RAM suficiente: ${ram_gb} GB"
    fi
    
    # Validar espacio en disco
    local disco_gb=$(obtener_espacio_disco_gb "/")
    registrar_info "Espacio en disco disponible: ${disco_gb} GB"
    
    if [[ ${disco_gb} -lt ${VIBE_MIN_DISK_GB} ]]; then
        registrar_error "Espacio en disco insuficiente: ${disco_gb} GB (mínimo: ${VIBE_MIN_DISK_GB} GB)"
        ((errores++))
    else
        registrar_exito "Espacio en disco suficiente: ${disco_gb} GB"
    fi
    
    # Validar CPUs
    local cpus=$(obtener_numero_cpus)
    registrar_info "CPUs detectadas: ${cpus}"
    
    if [[ ${cpus} -lt ${VIBE_MIN_CPU_CORES} ]]; then
        registrar_advertencia "CPUs por debajo del mínimo recomendado: ${cpus} (mínimo: ${VIBE_MIN_CPU_CORES})"
    else
        registrar_exito "CPUs suficientes: ${cpus}"
    fi
    
    if [[ ${errores} -gt 0 ]]; then
        registrar_error "El sistema NO cumple con los requisitos mínimos"
        return 1
    fi
    
    registrar_exito "El sistema cumple con los requisitos mínimos"
    return 0
}

# ============================================================================
# FUNCIÓN: validar_conectividad
# Propósito: Validar conectividad a Internet y repositorios
# Retorno: 0 si hay conectividad, 1 si no la hay
# ============================================================================
validar_conectividad() {
    registrar_info "Validando conectividad..."
    
    if ! verificar_conectividad_internet; then
        registrar_error "Se requiere conexión a Internet para la instalación"
        return 1
    fi
    
    # Verificar acceso a repositorios comunes
    local repos=("archive.ubuntu.com" "github.com" "pypi.org" "hub.docker.com")
    local errores=0
    
    for repo in "${repos[@]}"; do
        if ping -c 1 -W 2 "${repo}" &>/dev/null; then
            registrar_debug "Acceso a ${repo}: OK"
        else
            registrar_advertencia "No se pudo acceder a: ${repo}"
            ((errores++))
        fi
    done
    
    if [[ ${errores} -gt 2 ]]; then
        registrar_error "Problemas de conectividad detectados. Verifique su conexión."
        return 1
    fi
    
    registrar_exito "Conectividad verificada"
    return 0
}

# ============================================================================
# FUNCIÓN: validar_python_version
# Propósito: Validar versión de Python instalada
# Retorno: 0 si la versión es válida, 1 si no lo es
# ============================================================================
validar_python_version() {
    registrar_info "Validando versión de Python..."
    
    if ! verificar_comando_existe python3; then
        registrar_error "Python3 no está instalado"
        return 1
    fi
    
    local version_python=$(python3 --version 2>&1 | awk '{print $2}')
    registrar_info "Versión de Python detectada: ${version_python}"
    
    if comparar_versiones "${version_python}" "${VIBE_PYTHON_MIN_VERSION}"; then
        registrar_exito "Versión de Python válida: ${version_python}"
        return 0
    else
        registrar_error "Versión de Python insuficiente: ${version_python} (mínimo: ${VIBE_PYTHON_MIN_VERSION})"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: validar_docker_version
# Propósito: Validar versión de Docker instalada
# Retorno: 0 si la versión es válida, 1 si no lo es
# ============================================================================
validar_docker_version() {
    registrar_info "Validando versión de Docker..."
    
    if ! verificar_comando_existe docker; then
        registrar_advertencia "Docker no está instalado"
        return 1
    fi
    
    local version_docker=$(docker --version 2>&1 | grep -oP '(\d+\.\d+\.\d+)' | head -1)
    registrar_info "Versión de Docker detectada: ${version_docker}"
    
    if comparar_versiones "${version_docker}" "${VIBE_DOCKER_MIN_VERSION}"; then
        registrar_exito "Versión de Docker válida: ${version_docker}"
        return 0
    else
        registrar_error "Versión de Docker insuficiente: ${version_docker} (mínimo: ${VIBE_DOCKER_MIN_VERSION})"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: validar_docker_compose_version
# Propósito: Validar versión de Docker Compose instalada
# Retorno: 0 si la versión es válida, 1 si no lo es
# ============================================================================
validar_docker_compose_version() {
    registrar_info "Validando versión de Docker Compose..."
    
    if ! verificar_comando_existe docker-compose; then
        if ! docker compose version &>/dev/null; then
            registrar_advertencia "Docker Compose no está instalado"
            return 1
        fi
        local version_compose=$(docker compose version 2>&1 | grep -oP '(\d+\.\d+\.\d+)' | head -1)
    else
        local version_compose=$(docker-compose --version 2>&1 | grep -oP '(\d+\.\d+\.\d+)' | head -1)
    fi
    
    registrar_info "Versión de Docker Compose detectada: ${version_compose}"
    
    if comparar_versiones "${version_compose}" "${VIBE_DOCKER_COMPOSE_MIN_VERSION}"; then
        registrar_exito "Versión de Docker Compose válida: ${version_compose}"
        return 0
    else
        registrar_error "Versión de Docker Compose insuficiente: ${version_compose} (mínimo: ${VIBE_DOCKER_COMPOSE_MIN_VERSION})"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: validar_puertos_disponibles
# Propósito: Validar que los puertos necesarios estén disponibles
# Retorno: 0 si todos los puertos están disponibles, 1 si alguno está ocupado
# ============================================================================
validar_puertos_disponibles() {
    registrar_info "Validando disponibilidad de puertos..."
    
    local puertos=(
        "${VIBE_POSTGRES_PORT}"
        "${VIBE_REDIS_PORT}"
        "${VIBE_KAFKA_PORT}"
        "${VIBE_KAFKA_ZOOKEEPER_PORT}"
        "${VIBE_API_PORT}"
        "${VIBE_WEBSOCKET_PORT}"
        "${VIBE_ADMIN_PORT}"
    )
    
    local errores=0
    
    for puerto in "${puertos[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":${puerto} " || \
           ss -tuln 2>/dev/null | grep -q ":${puerto} "; then
            registrar_error "Puerto ${puerto} ya está en uso"
            ((errores++))
        else
            registrar_debug "Puerto ${puerto} disponible"
        fi
    done
    
    if [[ ${errores} -gt 0 ]]; then
        registrar_error "Hay puertos ocupados. Libere los puertos o modifique la configuración."
        return 1
    fi
    
    registrar_exito "Todos los puertos están disponibles"
    return 0
}

# ============================================================================
# FUNCIÓN: validar_privilegios
# Propósito: Validar que se tengan privilegios de root
# Retorno: 0 si tiene privilegios, 1 si no los tiene
# ============================================================================
validar_privilegios() {
    registrar_info "Validando privilegios de ejecución..."
    
    if verificar_root; then
        registrar_exito "Ejecutando con privilegios de root"
        return 0
    else
        registrar_error "Se requieren privilegios de root. Ejecute con sudo."
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: validar_configuracion
# Propósito: Validar que el archivo de configuración sea válido
# Retorno: 0 si la configuración es válida, 1 si no lo es
# ============================================================================
validar_configuracion() {
    registrar_info "Validando archivo de configuración..."
    
    local config_file="${SCRIPT_DIR}/../configuracion/stack-vibe.conf"
    
    if [[ ! -f "${config_file}" ]]; then
        registrar_error "No se encontró el archivo de configuración: ${config_file}"
        return 1
    fi
    
    # Verificar variables críticas
    local vars_criticas=(
        "VIBE_DIR_BASE"
        "VIBE_POSTGRES_DB"
        "VIBE_POSTGRES_USER"
        "VIBE_REDIS_PORT"
        "VIBE_API_PORT"
    )
    
    local errores=0
    
    for var in "${vars_criticas[@]}"; do
        if [[ -z "${!var}" ]]; then
            registrar_error "Variable crítica no definida: ${var}"
            ((errores++))
        else
            registrar_debug "Variable ${var} = ${!var}"
        fi
    done
    
    if [[ ${errores} -gt 0 ]]; then
        registrar_error "Configuración incompleta o inválida"
        return 1
    fi
    
    registrar_exito "Configuración validada correctamente"
    return 0
}

# ============================================================================
# FUNCIÓN: validar_servicio_activo
# Propósito: Validar que un servicio esté activo y funcionando
# Parámetros:
#   $1: Nombre del servicio
# Retorno: 0 si el servicio está activo, 1 si no lo está
# ============================================================================
validar_servicio_activo() {
    local servicio="$1"
    
    if systemctl is-active --quiet "${servicio}"; then
        registrar_exito "Servicio activo: ${servicio}"
        return 0
    else
        registrar_error "Servicio no activo: ${servicio}"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: validar_docker_funcionando
# Propósito: Validar que Docker esté funcionando correctamente
# Retorno: 0 si Docker funciona, 1 si no funciona
# ============================================================================
validar_docker_funcionando() {
    registrar_info "Validando funcionamiento de Docker..."
    
    if ! systemctl is-active --quiet docker; then
        registrar_error "El servicio Docker no está activo"
        return 1
    fi
    
    if docker ps &>/dev/null; then
        registrar_exito "Docker está funcionando correctamente"
        return 0
    else
        registrar_error "Docker no está funcionando correctamente"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: validar_instalacion_completa
# Propósito: Validar que todos los componentes estén instalados
# Retorno: 0 si todo está instalado, 1 si falta algo
# ============================================================================
validar_instalacion_completa() {
    registrar_info "Validando instalación completa de VibeVoice..."
    local errores=0
    
    # Validar directorios
    local dirs=(
        "${VIBE_DIR_BASE}"
        "${VIBE_DIR_DATOS}"
        "${VIBE_DIR_LOGS}"
        "${VIBE_DIR_CONFIG}"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ -d "${dir}" ]]; then
            registrar_debug "Directorio existe: ${dir}"
        else
            registrar_error "Directorio faltante: ${dir}"
            ((errores++))
        fi
    done
    
    # Validar comandos
    local comandos=("python3" "docker" "docker-compose")
    
    for cmd in "${comandos[@]}"; do
        if verificar_comando_existe "${cmd}"; then
            registrar_debug "Comando disponible: ${cmd}"
        else
            registrar_error "Comando faltante: ${cmd}"
            ((errores++))
        fi
    done
    
    if [[ ${errores} -gt 0 ]]; then
        registrar_error "Instalación incompleta. Faltan ${errores} componente(s)"
        return 1
    fi
    
    registrar_exito "Instalación completa validada"
    return 0
}

# ============================================================================
# EXPORTAR FUNCIONES
# ============================================================================
export -f validar_ubuntu_version
export -f validar_recursos_sistema
export -f validar_conectividad
export -f validar_python_version
export -f validar_docker_version
export -f validar_docker_compose_version
export -f validar_puertos_disponibles
export -f validar_privilegios
export -f validar_configuracion
export -f validar_servicio_activo
export -f validar_docker_funcionando
export -f validar_instalacion_completa

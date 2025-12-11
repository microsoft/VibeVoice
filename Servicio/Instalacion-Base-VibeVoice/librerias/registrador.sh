#!/bin/bash
# ============================================================================
# Archivo: registrador.sh
# Propósito: Sistema de bitácoras centralizado para trazabilidad completa
# Versión: 1.0.0
# Descripción: Proporciona funciones para registro de eventos, errores y auditoría
# ============================================================================

# Cargar configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configuracion/stack-vibe.conf" 2>/dev/null || true

# ============================================================================
# VARIABLES GLOBALES DE BITÁCORA
# ============================================================================
VIBE_LOG_DIR="${VIBE_LOG_DIR:-${VIBE_DIR_BASE}/logs}"
VIBE_LOG_FILE="${VIBE_LOG_DIR}/instalacion-$(date +%Y%m%d-%H%M%S).log"
VIBE_ERROR_LOG="${VIBE_LOG_DIR}/errores-$(date +%Y%m%d-%H%M%S).log"
VIBE_AUDIT_LOG="${VIBE_LOG_DIR}/auditoria-$(date +%Y%m%d-%H%M%S).log"

# Colores para salida en terminal
COLOR_RESET='\033[0m'
COLOR_ROJO='\033[0;31m'
COLOR_VERDE='\033[0;32m'
COLOR_AMARILLO='\033[0;33m'
COLOR_AZUL='\033[0;34m'
COLOR_MORADO='\033[0;35m'
COLOR_CYAN='\033[0;36m'

# ============================================================================
# FUNCIÓN: inicializar_bitacoras
# Propósito: Crear directorios de bitácoras y archivos iniciales
# ============================================================================
inicializar_bitacoras() {
    mkdir -p "${VIBE_LOG_DIR}" 2>/dev/null || {
        echo "ADVERTENCIA: No se pudo crear directorio de bitácoras en ${VIBE_LOG_DIR}"
        VIBE_LOG_DIR="./bitacoras"
        mkdir -p "${VIBE_LOG_DIR}"
        VIBE_LOG_FILE="${VIBE_LOG_DIR}/instalacion-$(date +%Y%m%d-%H%M%S).log"
        VIBE_ERROR_LOG="${VIBE_LOG_DIR}/errores-$(date +%Y%m%d-%H%M%S).log"
        VIBE_AUDIT_LOG="${VIBE_LOG_DIR}/auditoria-$(date +%Y%m%d-%H%M%S).log"
    }
    
    touch "${VIBE_LOG_FILE}" "${VIBE_ERROR_LOG}" "${VIBE_AUDIT_LOG}"
    
    registrar_info "===================================================="
    registrar_info "INICIO DE INSTALACIÓN: VibeVoice ${VIBE_VERSION}"
    registrar_info "Fecha: $(date '+%Y-%m-%d %H:%M:%S')"
    registrar_info "Usuario: $(whoami)"
    registrar_info "Hostname: $(hostname)"
    registrar_info "Sistema: $(uname -a)"
    registrar_info "===================================================="
}

# ============================================================================
# FUNCIÓN: registrar_mensaje
# Propósito: Función base para registro de mensajes
# Parámetros:
#   $1: Nivel (INFO, WARN, ERROR, DEBUG, SUCCESS)
#   $2: Mensaje
#   $3: Archivo de log (opcional)
# ============================================================================
registrar_mensaje() {
    local nivel="$1"
    local mensaje="$2"
    local archivo_log="${3:-${VIBE_LOG_FILE}}"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    local log_entry="[${timestamp}] [${nivel}] ${mensaje}"
    
    # Escribir a archivo de log
    echo "${log_entry}" >> "${archivo_log}"
    
    # Determinar color para terminal
    local color="${COLOR_RESET}"
    case "${nivel}" in
        ERROR)   color="${COLOR_ROJO}" ;;
        WARN)    color="${COLOR_AMARILLO}" ;;
        SUCCESS) color="${COLOR_VERDE}" ;;
        INFO)    color="${COLOR_AZUL}" ;;
        DEBUG)   color="${COLOR_CYAN}" ;;
        AUDIT)   color="${COLOR_MORADO}" ;;
    esac
    
    # Escribir a terminal con color
    echo -e "${color}[${nivel}]${COLOR_RESET} ${mensaje}"
}

# ============================================================================
# FUNCIONES DE REGISTRO POR NIVEL
# ============================================================================

registrar_info() {
    registrar_mensaje "INFO" "$1" "${VIBE_LOG_FILE}"
}

registrar_exito() {
    registrar_mensaje "SUCCESS" "$1" "${VIBE_LOG_FILE}"
}

registrar_advertencia() {
    registrar_mensaje "WARN" "$1" "${VIBE_LOG_FILE}"
}

registrar_error() {
    registrar_mensaje "ERROR" "$1" "${VIBE_ERROR_LOG}"
    registrar_mensaje "ERROR" "$1" "${VIBE_LOG_FILE}"
}

registrar_debug() {
    if [[ "${VIBE_LOG_LEVEL}" == "DEBUG" ]]; then
        registrar_mensaje "DEBUG" "$1" "${VIBE_LOG_FILE}"
    fi
}

registrar_auditoria() {
    registrar_mensaje "AUDIT" "$1" "${VIBE_AUDIT_LOG}"
    registrar_mensaje "AUDIT" "$1" "${VIBE_LOG_FILE}"
}

# ============================================================================
# FUNCIÓN: registrar_comando
# Propósito: Registrar ejecución de comandos y sus resultados
# Parámetros:
#   $1: Comando ejecutado
#   $2: Código de salida
#   $3: Salida del comando (opcional)
# ============================================================================
registrar_comando() {
    local comando="$1"
    local codigo_salida="$2"
    local salida="${3:-}"
    
    registrar_auditoria "Comando ejecutado: ${comando}"
    registrar_auditoria "Código de salida: ${codigo_salida}"
    
    if [[ ${codigo_salida} -eq 0 ]]; then
        registrar_exito "Comando exitoso: ${comando}"
    else
        registrar_error "Comando falló (código ${codigo_salida}): ${comando}"
        if [[ -n "${salida}" ]]; then
            registrar_error "Salida: ${salida}"
        fi
    fi
}

# ============================================================================
# FUNCIÓN: registrar_inicio_modulo
# Propósito: Marcar inicio de un módulo de instalación
# Parámetros:
#   $1: Nombre del módulo
# ============================================================================
registrar_inicio_modulo() {
    local modulo="$1"
    echo ""
    registrar_info "╔══════════════════════════════════════════════════════════════╗"
    registrar_info "║ INICIANDO MÓDULO: ${modulo}"
    registrar_info "╚══════════════════════════════════════════════════════════════╝"
    registrar_auditoria "Inicio de módulo: ${modulo}"
}

# ============================================================================
# FUNCIÓN: registrar_fin_modulo
# Propósito: Marcar finalización exitosa de un módulo
# Parámetros:
#   $1: Nombre del módulo
# ============================================================================
registrar_fin_modulo() {
    local modulo="$1"
    registrar_exito "Módulo completado exitosamente: ${modulo}"
    registrar_auditoria "Fin de módulo: ${modulo}"
    echo ""
}

# ============================================================================
# FUNCIÓN: registrar_error_fatal
# Propósito: Registrar error fatal y terminar instalación
# Parámetros:
#   $1: Mensaje de error
#   $2: Código de salida (opcional, por defecto 1)
# ============================================================================
registrar_error_fatal() {
    local mensaje="$1"
    local codigo="${2:-1}"
    
    echo ""
    registrar_error "╔══════════════════════════════════════════════════════════════╗"
    registrar_error "║ ERROR FATAL: ${mensaje}"
    registrar_error "╚══════════════════════════════════════════════════════════════╝"
    registrar_auditoria "Error fatal: ${mensaje} (código: ${codigo})"
    registrar_info "Instalación abortada. Revise los logs en: ${VIBE_LOG_DIR}"
    exit "${codigo}"
}

# ============================================================================
# FUNCIÓN: obtener_estadisticas_bitacoras
# Propósito: Generar resumen de bitácoras
# ============================================================================
obtener_estadisticas_bitacoras() {
    echo ""
    registrar_info "═══════════════════════════════════════════════════════════════"
    registrar_info "ESTADÍSTICAS DE INSTALACIÓN"
    registrar_info "═══════════════════════════════════════════════════════════════"
    
    if [[ -f "${VIBE_LOG_FILE}" ]]; then
        local total_info=$(grep -c "\[INFO\]" "${VIBE_LOG_FILE}" 2>/dev/null || echo "0")
        local total_success=$(grep -c "\[SUCCESS\]" "${VIBE_LOG_FILE}" 2>/dev/null || echo "0")
        local total_warn=$(grep -c "\[WARN\]" "${VIBE_LOG_FILE}" 2>/dev/null || echo "0")
        local total_error=$(grep -c "\[ERROR\]" "${VIBE_LOG_FILE}" 2>/dev/null || echo "0")
        
        registrar_info "Mensajes informativos: ${total_info}"
        registrar_info "Operaciones exitosas: ${total_success}"
        registrar_info "Advertencias: ${total_warn}"
        registrar_info "Errores: ${total_error}"
        registrar_info "Archivo de log principal: ${VIBE_LOG_FILE}"
        registrar_info "Archivo de errores: ${VIBE_ERROR_LOG}"
        registrar_info "Archivo de auditoría: ${VIBE_AUDIT_LOG}"
    fi
    
    registrar_info "═══════════════════════════════════════════════════════════════"
}

# ============================================================================
# EXPORTAR FUNCIONES
# ============================================================================
export -f inicializar_bitacoras
export -f registrar_mensaje
export -f registrar_info
export -f registrar_exito
export -f registrar_advertencia
export -f registrar_error
export -f registrar_debug
export -f registrar_auditoria
export -f registrar_comando
export -f registrar_inicio_modulo
export -f registrar_fin_modulo
export -f registrar_error_fatal
export -f obtener_estadisticas_bitacoras

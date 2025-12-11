#!/bin/bash
# ============================================================================
# Módulo: 01-verificacion-sistema.sh
# Propósito: Verificar requisitos del sistema antes de la instalación
# Versión: 1.0.0
# Descripción: Valida Ubuntu, recursos, conectividad y privilegios
# ============================================================================

set -euo pipefail

# Cargar dependencias
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configuracion/stack-vibe.conf"
source "${SCRIPT_DIR}/../librerias/registrador.sh"
source "${SCRIPT_DIR}/../librerias/ayudante.sh"
source "${SCRIPT_DIR}/../librerias/validador.sh"

# ============================================================================
# FUNCIÓN PRINCIPAL: verificar_sistema
# ============================================================================
verificar_sistema() {
    registrar_inicio_modulo "Verificación de Sistema"
    
    local errores=0
    
    # Validar privilegios de root
    if ! validar_privilegios; then
        registrar_error_fatal "Se requieren privilegios de root para ejecutar este instalador" 2
    fi
    
    # Validar versión de Ubuntu
    if ! validar_ubuntu_version; then
        registrar_error_fatal "Sistema operativo no soportado" 3
    fi
    
    # Validar recursos del sistema
    if ! validar_recursos_sistema; then
        registrar_error "Recursos del sistema insuficientes"
        ((errores++))
    fi
    
    # Validar conectividad
    if ! validar_conectividad; then
        registrar_error "Problemas de conectividad detectados"
        ((errores++))
    fi
    
    # Validar configuración
    if ! validar_configuracion; then
        registrar_error_fatal "Configuración inválida" 4
    fi
    
    # Validar puertos disponibles
    if ! validar_puertos_disponibles; then
        registrar_advertencia "Algunos puertos están ocupados. Esto puede causar conflictos."
        if ! solicitar_confirmacion "¿Desea continuar de todos modos?"; then
            registrar_error_fatal "Instalación cancelada por el usuario" 5
        fi
    fi
    
    # Mostrar resumen del sistema
    echo ""
    registrar_info "═══════════════════════════════════════════════════════════════"
    registrar_info "RESUMEN DEL SISTEMA"
    registrar_info "═══════════════════════════════════════════════════════════════"
    registrar_info "Sistema Operativo: $(lsb_release -d | cut -f2)"
    registrar_info "Kernel: $(uname -r)"
    registrar_info "RAM: $(obtener_memoria_total_gb) GB"
    registrar_info "CPUs: $(obtener_numero_cpus)"
    registrar_info "Espacio en Disco: $(obtener_espacio_disco_gb /) GB"
    registrar_info "Usuario: $(whoami)"
    registrar_info "Hostname: $(hostname)"
    registrar_info "═══════════════════════════════════════════════════════════════"
    echo ""
    
    if [[ ${errores} -gt 0 ]]; then
        if ! solicitar_confirmacion "Se detectaron ${errores} advertencia(s). ¿Desea continuar?"; then
            registrar_error_fatal "Instalación cancelada por el usuario" 6
        fi
    fi

    # Intentar crear el usuario de servicio temprano para evitar problemas de permisos
    if ! crear_usuario_sistema "${VIBE_SERVICE_USER}" "${VIBE_DIR_BASE}"; then
        registrar_advertencia "No se pudo crear el usuario de servicio ${VIBE_SERVICE_USER}. Algunos pasos pueden omitir chown." 
    fi
    
    registrar_fin_modulo "Verificación de Sistema"
    return 0
}

# ============================================================================
# EJECUCIÓN
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    verificar_sistema
fi

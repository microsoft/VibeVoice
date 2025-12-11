#!/bin/bash
# ============================================================================
# Script: probar-modulo.sh
# Propósito: Probar módulos de instalación individualmente
# Versión: 1.0.0
# Descripción: Permite ejecutar y probar módulos de instalación por separado
# ============================================================================

set -euo pipefail

# Cargar dependencias
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configuracion/stack-vibe.conf"
source "${SCRIPT_DIR}/../librerias/registrador.sh"
source "${SCRIPT_DIR}/../librerias/ayudante.sh"
source "${SCRIPT_DIR}/../librerias/validador.sh"

# ============================================================================
# FUNCIÓN: mostrar_ayuda
# ============================================================================
mostrar_ayuda() {
    cat <<EOF

PROBADOR DE MÓDULOS - VIBEVOICE

USO:
    sudo ./probar-modulo.sh [NÚMERO_MÓDULO]

MÓDULOS DISPONIBLES:
    01  Verificación del Sistema
    02  Instalación de Python
    03  Instalación de Docker
    04  Configuración de Servicios de Datos
    05  Configuración de Docker Compose
    06  Configuración de Servicio Systemd

EJEMPLOS:
    # Probar módulo de verificación
    sudo ./probar-modulo.sh 01

    # Probar módulo de Python
    sudo ./probar-modulo.sh 02

    # Listar módulos disponibles
    ./probar-modulo.sh --list

OPCIONES:
    -h, --help      Mostrar esta ayuda
    -l, --list      Listar módulos disponibles

EOF
}

# ============================================================================
# FUNCIÓN: listar_modulos
# ============================================================================
listar_modulos() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "MÓDULOS DE INSTALACIÓN DISPONIBLES"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""
    
    local modulos_dir="${SCRIPT_DIR}/../modulos"
    
    for modulo in "${modulos_dir}"/*.sh; do
        if [[ -f "${modulo}" ]]; then
            local nombre=$(basename "${modulo}")
            local numero=$(echo "${nombre}" | grep -oP '^\d+')
            local descripcion=$(grep -m 1 "^# Módulo:" "${modulo}" | cut -d':' -f2- | xargs)
            
            if [[ -z "${descripcion}" ]]; then
                descripcion=$(grep -m 1 "^# Propósito:" "${modulo}" | cut -d':' -f2- | xargs)
            fi
            
            printf "  %s  %-40s  %s\n" "${numero}" "${nombre}" "${descripcion}"
        fi
    done
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""
}

# ============================================================================
# FUNCIÓN: probar_modulo
# ============================================================================
probar_modulo() {
    local numero="$1"
    
    # Formatear número con ceros a la izquierda
    numero=$(printf "%02d" "${numero}")
    
    local modulo_script="${SCRIPT_DIR}/../modulos/${numero}-*.sh"
    
    # Buscar archivo del módulo
    if ! compgen -G "${modulo_script}" > /dev/null; then
        registrar_error "No se encontró el módulo: ${numero}"
        echo ""
        echo "Use './probar-modulo.sh --list' para ver los módulos disponibles"
        echo ""
        return 1
    fi
    
    # Obtener nombre exacto del archivo
    modulo_script=$(compgen -G "${modulo_script}" | head -1)
    
    if [[ ! -f "${modulo_script}" ]]; then
        registrar_error "El módulo no existe: ${modulo_script}"
        return 1
    fi
    
    # Verificar privilegios de root
    if [[ $EUID -ne 0 ]]; then
        echo ""
        echo "ERROR: Este script debe ejecutarse con privilegios de root"
        echo "Por favor, ejecute: sudo $0 ${numero}"
        echo ""
        return 1
    fi
    
    # Inicializar bitácoras
    inicializar_bitacoras
    
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║                  PRUEBA DE MÓDULO DE INSTALACIÓN                    ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    local nombre_modulo=$(basename "${modulo_script}")
    registrar_info "Módulo a ejecutar: ${nombre_modulo}"
    
    # Obtener descripción del módulo
    local descripcion=$(grep -m 1 "^# Propósito:" "${modulo_script}" | cut -d':' -f2- | xargs)
    if [[ -n "${descripcion}" ]]; then
        registrar_info "Descripción: ${descripcion}"
    fi
    
    echo ""
    
    if ! solicitar_confirmacion "¿Desea ejecutar este módulo?"; then
        registrar_info "Ejecución cancelada por el usuario"
        return 0
    fi
    
    # Ejecutar módulo
    local inicio=$(date +%s)
    
    echo ""
    registrar_info "════════════════════════════════════════════════════════════════"
    registrar_info "INICIANDO EJECUCIÓN DEL MÓDULO"
    registrar_info "════════════════════════════════════════════════════════════════"
    echo ""
    
    if bash "${modulo_script}"; then
        local fin=$(date +%s)
        local duracion=$((fin - inicio))
        
        echo ""
        echo "╔══════════════════════════════════════════════════════════════════════╗"
        echo "║                                                                      ║"
        echo "║              ✓ MÓDULO EJECUTADO EXITOSAMENTE                        ║"
        echo "║                                                                      ║"
        echo "╚══════════════════════════════════════════════════════════════════════╝"
        echo ""
        registrar_exito "Módulo completado en ${duracion} segundos"
        return 0
    else
        local codigo=$?
        local fin=$(date +%s)
        local duracion=$((fin - inicio))
        
        echo ""
        echo "╔══════════════════════════════════════════════════════════════════════╗"
        echo "║                                                                      ║"
        echo "║              ✗ ERROR EN LA EJECUCIÓN DEL MÓDULO                     ║"
        echo "║                                                                      ║"
        echo "╚══════════════════════════════════════════════════════════════════════╝"
        echo ""
        registrar_error "Módulo falló después de ${duracion} segundos (código: ${codigo})"
        registrar_info "Revise los logs para más detalles"
        return ${codigo}
    fi
}

# ============================================================================
# FUNCIÓN: main
# ============================================================================
main() {
    # Procesar argumentos
    if [[ $# -eq 0 ]]; then
        mostrar_ayuda
        return 1
    fi
    
    case "$1" in
        -h|--help)
            mostrar_ayuda
            return 0
            ;;
        -l|--list)
            listar_modulos
            return 0
            ;;
        [0-9]|[0-9][0-9])
            probar_modulo "$1"
            return $?
            ;;
        *)
            echo "Argumento inválido: $1"
            mostrar_ayuda
            return 1
            ;;
    esac
}

# ============================================================================
# EJECUCIÓN
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

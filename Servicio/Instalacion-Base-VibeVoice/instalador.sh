#!/bin/bash
# ============================================================================
# Instalador Principal: VibeVoice
# Propósito: Orquestar la instalación completa de VibeVoice en Ubuntu 22.04/24.04
# Versión: 1.0.0
# Descripción: Instalador modular, idempotente y con trazabilidad completa
# ============================================================================

set -euo pipefail

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================
INSTALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER_VERSION="1.0.0"

# Cargar configuración y librerías
source "${INSTALLER_DIR}/configuracion/stack-vibe.conf"
source "${INSTALLER_DIR}/librerias/registrador.sh"
source "${INSTALLER_DIR}/librerias/ayudante.sh"
source "${INSTALLER_DIR}/librerias/validador.sh"

# ============================================================================
# FUNCIONES DE BANNER
# ============================================================================
mostrar_banner() {
    clear
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║                    INSTALADOR VIBEVOICE v${INSTALLER_VERSION}                    ║"
    echo "║                                                                      ║"
    echo "║            Sistema de Transcripción y Análisis de Voz con IA        ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Versión del Sistema:     ${VIBE_VERSION}"
    echo "  Plataforma:              Ubuntu 22.04 / 24.04"
    echo "  Tipo de Instalación:     Completa (Base + Servicios + Systemd)"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
}

mostrar_resumen_instalacion() {
    echo ""
    registrar_info "╔══════════════════════════════════════════════════════════════╗"
    registrar_info "║           RESUMEN DE COMPONENTES A INSTALAR                  ║"
    registrar_info "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    registrar_info "Módulos de instalación:"
    registrar_info "  1. Verificación del sistema (Ubuntu, recursos, conectividad)"
    registrar_info "  2. Python ${VIBE_PYTHON_VERSION} + pip + virtualenv + dependencias"
    registrar_info "  3. Docker Engine + Docker Compose"
    registrar_info "  4. Servicios de datos:"
    registrar_info "     • PostgreSQL ${VIBE_POSTGRES_VERSION} (Base de datos)"
    registrar_info "     • Redis ${VIBE_REDIS_VERSION} (Caché y colas)"
    registrar_info "     • Kafka ${VIBE_KAFKA_VERSION} + Zookeeper (Mensajería)"
    registrar_info "  5. Docker Compose (Orquestación de servicios)"
    registrar_info "  6. Servicio Systemd (Inicio automático)"
    echo ""
    registrar_info "Directorios de instalación:"
    registrar_info "  • Base:      ${VIBE_DIR_BASE}"
    registrar_info "  • Datos:     ${VIBE_DIR_DATOS}"
    registrar_info "  • Logs:      ${VIBE_DIR_LOGS}"
    registrar_info "  • Config:    ${VIBE_DIR_CONFIG}"
    registrar_info "  • Backups:   ${VIBE_DIR_BACKUPS}"
    echo ""
    registrar_info "═══════════════════════════════════════════════════════════════"
    echo ""
}

# ============================================================================
# FUNCIÓN: ejecutar_modulo
# Propósito: Ejecutar un módulo de instalación con manejo de errores
# Parámetros:
#   $1: Número del módulo
#   $2: Nombre del script del módulo
#   $3: Descripción del módulo
# ============================================================================
ejecutar_modulo() {
    local numero="$1"
    local script="$2"
    local descripcion="$3"
    local script_path="${INSTALLER_DIR}/modulos/${script}"
    
    echo ""
    registrar_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    registrar_info "MÓDULO ${numero}: ${descripcion}"
    registrar_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [[ ! -f "${script_path}" ]]; then
        registrar_error_fatal "No se encontró el módulo: ${script_path}" 10
    fi
    
    # Ejecutar módulo
    if bash "${script_path}"; then
        registrar_exito "✓ Módulo ${numero} completado exitosamente"
        return 0
    else
        local codigo=$?
        registrar_error "✗ Error en módulo ${numero}: ${descripcion}"
        registrar_error_fatal "Instalación abortada en módulo ${numero}" ${codigo}
        return ${codigo}
    fi
}

# ============================================================================
# FUNCIÓN: verificar_prerequisitos
# ============================================================================
verificar_prerequisitos() {
    registrar_info "Verificando prerequisitos de instalación..."
    
    # Verificar que se ejecuta como root
    if [[ $EUID -ne 0 ]]; then
        echo ""
        echo "ERROR: Este instalador debe ejecutarse con privilegios de root"
        echo "Por favor, ejecute: sudo $0"
        echo ""
        exit 1
    fi
    
    # Verificar que todos los módulos existen
    local modulos=(
        "01-verificacion-sistema.sh"
        "02-python.sh"
        "03-docker.sh"
        "04-servicios-datos.sh"
        "05-docker-compose.sh"
        "06-systemd-service.sh"
    )
    
    for modulo in "${modulos[@]}"; do
        if [[ ! -f "${INSTALLER_DIR}/modulos/${modulo}" ]]; then
            echo "ERROR: Falta el módulo: ${modulo}"
            exit 1
        fi
    done
    
    registrar_debug "Prerequisitos verificados correctamente"
}

# ============================================================================
# FUNCIÓN: instalacion_principal
# ============================================================================
instalacion_principal() {
    local inicio=$(date +%s)
    
    # Inicializar sistema de bitácoras
    inicializar_bitacoras
    
    # Mostrar banner
    mostrar_banner
    
    # Mostrar resumen
    mostrar_resumen_instalacion
    
    # Solicitar confirmación
    if ! solicitar_confirmacion "¿Desea continuar con la instalación?"; then
        registrar_info "Instalación cancelada por el usuario"
        exit 0
    fi
    
    # Ejecutar módulos en orden
    ejecutar_modulo "01" "01-verificacion-sistema.sh" "Verificación del Sistema"
    ejecutar_modulo "02" "02-python.sh" "Instalación de Python"
    ejecutar_modulo "03" "03-docker.sh" "Instalación de Docker"
    ejecutar_modulo "04" "04-servicios-datos.sh" "Configuración de Servicios de Datos"
    ejecutar_modulo "05" "05-docker-compose.sh" "Configuración de Docker Compose"
    ejecutar_modulo "06" "06-systemd-service.sh" "Configuración de Servicio Systemd"
    
    # Calcular tiempo de instalación
    local fin=$(date +%s)
    local duracion=$((fin - inicio))
    local minutos=$((duracion / 60))
    local segundos=$((duracion % 60))
    
    # Mostrar resultado final
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║              ✓ INSTALACIÓN COMPLETADA EXITOSAMENTE                  ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    registrar_exito "Instalación de VibeVoice completada exitosamente"
    registrar_info "Tiempo total de instalación: ${minutos} minutos ${segundos} segundos"
    echo ""
    
    # Mostrar instrucciones post-instalación
    mostrar_instrucciones_postinstalacion
    
    # Mostrar estadísticas de bitácoras
    obtener_estadisticas_bitacoras
}

# ============================================================================
# FUNCIÓN: mostrar_instrucciones_postinstalacion
# ============================================================================
mostrar_instrucciones_postinstalacion() {
    echo ""
    registrar_info "╔══════════════════════════════════════════════════════════════╗"
    registrar_info "║           INSTRUCCIONES POST-INSTALACIÓN                     ║"
    registrar_info "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    registrar_info "1. Iniciar servicios de VibeVoice:"
    registrar_info "   sudo systemctl start ${VIBE_SERVICE_NAME}"
    echo ""
    registrar_info "2. Verificar estado del servicio:"
    registrar_info "   sudo systemctl status ${VIBE_SERVICE_NAME}"
    echo ""
    registrar_info "3. Ver logs en tiempo real:"
    registrar_info "   sudo journalctl -u ${VIBE_SERVICE_NAME} -f"
    echo ""
    registrar_info "4. Validar instalación:"
    registrar_info "   cd ${INSTALLER_DIR}"
    registrar_info "   sudo ./pruebas/validar-instalacion.sh"
    echo ""
    registrar_info "5. Acceder a los servicios:"
    registrar_info "   • API:         http://localhost:${VIBE_API_PORT}"
    registrar_info "   • PostgreSQL:  localhost:${VIBE_POSTGRES_PORT}"
    registrar_info "   • Redis:       localhost:${VIBE_REDIS_PORT}"
    registrar_info "   • Kafka:       localhost:${VIBE_KAFKA_PORT}"
    echo ""
    registrar_advertencia "═══════════════════════════════════════════════════════════════"
    registrar_advertencia "IMPORTANTE - SEGURIDAD:"
    registrar_advertencia "═══════════════════════════════════════════════════════════════"
    registrar_advertencia "1. Cambiar contraseñas por defecto en: ${VIBE_DIR_CONFIG}/.env"
    registrar_advertencia "2. Configurar firewall para limitar acceso a puertos"
    registrar_advertencia "3. Habilitar SSL/TLS para conexiones remotas"
    registrar_advertencia "4. Implementar secret manager en producción"
    registrar_advertencia "5. Configurar backups automáticos de datos"
    registrar_advertencia "═══════════════════════════════════════════════════════════════"
    echo ""
    registrar_info "Para más información, consulte:"
    registrar_info "  • ${INSTALLER_DIR}/LEEME.md"
    registrar_info "  • ${INSTALLER_DIR}/GLOSARIO.md"
    registrar_info "  • https://github.com/jhoavera/VibeVoice"
    echo ""
}

# ============================================================================
# FUNCIÓN: mostrar_ayuda
# ============================================================================
mostrar_ayuda() {
    cat <<EOF

INSTALADOR DE VIBEVOICE v${INSTALLER_VERSION}

USO:
    sudo ./instalador.sh [OPCIONES]

OPCIONES:
    -h, --help              Mostrar esta ayuda
    -v, --version           Mostrar versión del instalador
    -c, --check             Solo verificar requisitos sin instalar
    --skip-confirmation     Omitir confirmación inicial

DESCRIPCIÓN:
    Instalador modular e idempotente para VibeVoice en Ubuntu 22.04/24.04
    Instala y configura todos los componentes necesarios:
    - Python ${VIBE_PYTHON_VERSION} y dependencias
    - Docker y Docker Compose
    - PostgreSQL, Redis, Kafka
    - Servicio systemd para inicio automático

EJEMPLOS:
    # Instalación normal
    sudo ./instalador.sh

    # Solo verificar requisitos
    sudo ./instalador.sh --check

    # Instalación sin confirmación (modo automatizado)
    sudo ./instalador.sh --skip-confirmation

ARCHIVOS DE LOG:
    Los logs de instalación se guardan en: ${VIBE_DIR_BASE}/logs/
    o en ./bitacoras/ si no se puede acceder al directorio anterior

SOPORTE:
    https://github.com/jhoavera/VibeVoice/issues

EOF
}

# ============================================================================
# PROCESAMIENTO DE ARGUMENTOS
# ============================================================================
procesar_argumentos() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                mostrar_ayuda
                exit 0
                ;;
            -v|--version)
                echo "Instalador VibeVoice v${INSTALLER_VERSION}"
                exit 0
                ;;
            -c|--check)
                inicializar_bitacoras
                verificar_prerequisitos
                bash "${INSTALLER_DIR}/modulos/01-verificacion-sistema.sh"
                echo ""
                echo "Verificación completada. Sistema listo para instalación."
                exit 0
                ;;
            --skip-confirmation)
                SKIP_CONFIRMATION=true
                shift
                ;;
            *)
                echo "Opción desconocida: $1"
                echo "Use --help para ver las opciones disponibles"
                exit 1
                ;;
        esac
        shift
    done
}

# ============================================================================
# MANEJO DE SEÑALES
# ============================================================================
trap 'registrar_error "Instalación interrumpida por el usuario"; exit 130' INT TERM

# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================
main() {
    # Verificar prerequisitos básicos
    verificar_prerequisitos
    
    # Procesar argumentos de línea de comandos
    procesar_argumentos "$@"
    
    # Ejecutar instalación principal
    instalacion_principal
    
    exit 0
}

# Ejecutar main
main "$@"

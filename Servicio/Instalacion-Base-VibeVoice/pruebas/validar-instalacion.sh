#!/bin/bash
# ============================================================================
# Script: validar-instalacion.sh
# Propósito: Validar que la instalación de VibeVoice sea correcta
# Versión: 1.0.0
# Descripción: Ejecuta pruebas de validación post-instalación
# ============================================================================

set -euo pipefail

# Cargar dependencias
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configuracion/stack-vibe.conf"
source "${SCRIPT_DIR}/../librerias/registrador.sh"
source "${SCRIPT_DIR}/../librerias/ayudante.sh"
source "${SCRIPT_DIR}/../librerias/validador.sh"

# ============================================================================
# VARIABLES GLOBALES
# ============================================================================
TOTAL_PRUEBAS=0
PRUEBAS_EXITOSAS=0
PRUEBAS_FALLIDAS=0

# ============================================================================
# FUNCIÓN: ejecutar_prueba
# ============================================================================
ejecutar_prueba() {
    local nombre="$1"
    local comando="$2"
    
    ((TOTAL_PRUEBAS++))
    
    echo -n "  [${TOTAL_PRUEBAS}] ${nombre}... "
    
    if eval "${comando}" &>/dev/null; then
        echo -e "${COLOR_VERDE}✓ OK${COLOR_RESET}"
        ((PRUEBAS_EXITOSAS++))
        return 0
    else
        echo -e "${COLOR_ROJO}✗ FALLO${COLOR_RESET}"
        ((PRUEBAS_FALLIDAS++))
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: validar_instalacion_completa
# ============================================================================
validar_instalacion_completa() {
    inicializar_bitacoras
    
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║              VALIDACIÓN DE INSTALACIÓN - VIBEVOICE                  ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    registrar_info "Iniciando validación de instalación..."
    echo ""
    
    # ========================================================================
    # SECCIÓN 1: Comandos del Sistema
    # ========================================================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1. COMANDOS DEL SISTEMA"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    ejecutar_prueba "Python 3 instalado" "command -v python3"
    ejecutar_prueba "pip instalado" "command -v pip3"
    ejecutar_prueba "Docker instalado" "command -v docker"
    ejecutar_prueba "Docker Compose instalado" "command -v docker-compose"
    ejecutar_prueba "curl instalado" "command -v curl"
    ejecutar_prueba "git instalado" "command -v git"
    
    echo ""
    
    # ========================================================================
    # SECCIÓN 2: Versiones
    # ========================================================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "2. VERSIONES DE SOFTWARE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    echo "  Python:          $(python3 --version 2>&1 | awk '{print $2}')"
    echo "  pip:             $(pip3 --version 2>&1 | awk '{print $2}')"
    echo "  Docker:          $(docker --version 2>&1 | awk '{print $3}' | sed 's/,//')"
    echo "  Docker Compose:  $(docker-compose --version 2>&1 | awk '{print $4}')"
    
    echo ""
    
    # ========================================================================
    # SECCIÓN 3: Directorios
    # ========================================================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "3. DIRECTORIOS DE INSTALACIÓN"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    ejecutar_prueba "Directorio base existe" "test -d ${VIBE_DIR_BASE}"
    ejecutar_prueba "Directorio de datos existe" "test -d ${VIBE_DIR_DATOS}"
    ejecutar_prueba "Directorio de logs existe" "test -d ${VIBE_DIR_LOGS}"
    ejecutar_prueba "Directorio de config existe" "test -d ${VIBE_DIR_CONFIG}"
    ejecutar_prueba "Directorio de backups existe" "test -d ${VIBE_DIR_BACKUPS}"
    
    echo ""
    
    # ========================================================================
    # SECCIÓN 4: Servicios Docker
    # ========================================================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "4. SERVICIOS DOCKER"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    ejecutar_prueba "Servicio Docker activo" "systemctl is-active docker"
    ejecutar_prueba "Red Docker existe" "docker network ls | grep -q ${VIBE_DOCKER_NETWORK}"
    
    # Verificar contenedores si están corriendo
    if docker ps &>/dev/null; then
        ejecutar_prueba "Docker funcional" "docker ps"
        
        # Opcional: verificar contenedores específicos si existen
        if docker ps -a | grep -q "vibevoice-postgres"; then
            ejecutar_prueba "Contenedor PostgreSQL existe" "docker ps -a | grep -q vibevoice-postgres"
        fi
        
        if docker ps -a | grep -q "vibevoice-redis"; then
            ejecutar_prueba "Contenedor Redis existe" "docker ps -a | grep -q vibevoice-redis"
        fi
        
        if docker ps -a | grep -q "vibevoice-kafka"; then
            ejecutar_prueba "Contenedor Kafka existe" "docker ps -a | grep -q vibevoice-kafka"
        fi
    fi
    
    echo ""
    
    # ========================================================================
    # SECCIÓN 5: Archivos de Configuración
    # ========================================================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "5. ARCHIVOS DE CONFIGURACIÓN"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    ejecutar_prueba "stack-vibe.conf existe" "test -f ${SCRIPT_DIR}/../configuracion/stack-vibe.conf"
    
    if [[ -f "${VIBE_DIR_CONFIG}/docker-compose.yml" ]]; then
        ejecutar_prueba "docker-compose.yml existe" "test -f ${VIBE_DIR_CONFIG}/docker-compose.yml"
    fi
    
    if [[ -f "${VIBE_DIR_CONFIG}/.env" ]]; then
        ejecutar_prueba ".env existe" "test -f ${VIBE_DIR_CONFIG}/.env"
    fi
    
    echo ""
    
    # ========================================================================
    # SECCIÓN 6: Servicio Systemd
    # ========================================================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "6. SERVICIO SYSTEMD"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [[ -f "/etc/systemd/system/${VIBE_SERVICE_NAME}.service" ]]; then
        ejecutar_prueba "Archivo de servicio existe" "test -f /etc/systemd/system/${VIBE_SERVICE_NAME}.service"
        ejecutar_prueba "Servicio habilitado" "systemctl is-enabled ${VIBE_SERVICE_NAME}"
    else
        echo "  Servicio systemd no configurado (opcional)"
    fi
    
    echo ""
    
    # ========================================================================
    # SECCIÓN 7: Conectividad de Puertos
    # ========================================================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "7. CONECTIVIDAD DE PUERTOS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Verificar si los puertos están en uso (si los servicios están corriendo)
    for puerto in ${VIBE_POSTGRES_PORT} ${VIBE_REDIS_PORT} ${VIBE_KAFKA_PORT}; do
        if netstat -tuln 2>/dev/null | grep -q ":${puerto} " || ss -tuln 2>/dev/null | grep -q ":${puerto} "; then
            echo -e "  Puerto ${puerto}:  ${COLOR_VERDE}✓ EN USO${COLOR_RESET}"
        else
            echo -e "  Puerto ${puerto}:  ${COLOR_AMARILLO}○ Disponible (servicio no iniciado)${COLOR_RESET}"
        fi
    done
    
    echo ""
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                       RESUMEN DE VALIDACIÓN                          ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Total de pruebas:     ${TOTAL_PRUEBAS}"
    echo -e "  Pruebas exitosas:     ${COLOR_VERDE}${PRUEBAS_EXITOSAS}${COLOR_RESET}"
    
    if [[ ${PRUEBAS_FALLIDAS} -gt 0 ]]; then
        echo -e "  Pruebas fallidas:     ${COLOR_ROJO}${PRUEBAS_FALLIDAS}${COLOR_RESET}"
    else
        echo -e "  Pruebas fallidas:     ${COLOR_VERDE}${PRUEBAS_FALLIDAS}${COLOR_RESET}"
    fi
    
    local porcentaje=$((PRUEBAS_EXITOSAS * 100 / TOTAL_PRUEBAS))
    echo "  Porcentaje de éxito:  ${porcentaje}%"
    echo ""
    
    if [[ ${PRUEBAS_FALLIDAS} -eq 0 ]]; then
        echo -e "${COLOR_VERDE}╔══════════════════════════════════════════════════════════════════════╗${COLOR_RESET}"
        echo -e "${COLOR_VERDE}║                                                                      ║${COLOR_RESET}"
        echo -e "${COLOR_VERDE}║              ✓ INSTALACIÓN VALIDADA EXITOSAMENTE                    ║${COLOR_RESET}"
        echo -e "${COLOR_VERDE}║                                                                      ║${COLOR_RESET}"
        echo -e "${COLOR_VERDE}╚══════════════════════════════════════════════════════════════════════╝${COLOR_RESET}"
        echo ""
        registrar_exito "Validación completada: Todas las pruebas pasaron"
        return 0
    else
        echo -e "${COLOR_AMARILLO}╔══════════════════════════════════════════════════════════════════════╗${COLOR_RESET}"
        echo -e "${COLOR_AMARILLO}║                                                                      ║${COLOR_RESET}"
        echo -e "${COLOR_AMARILLO}║              ⚠ INSTALACIÓN PARCIALMENTE VALIDADA                    ║${COLOR_RESET}"
        echo -e "${COLOR_AMARILLO}║                                                                      ║${COLOR_RESET}"
        echo -e "${COLOR_AMARILLO}╚══════════════════════════════════════════════════════════════════════╝${COLOR_RESET}"
        echo ""
        registrar_advertencia "Validación completada: ${PRUEBAS_FALLIDAS} prueba(s) fallaron"
        echo "  Revise los componentes que fallaron e intente reinstalarlos."
        echo ""
        return 1
    fi
}

# ============================================================================
# EJECUCIÓN
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    validar_instalacion_completa
fi

#!/bin/bash
# ============================================================================
# Módulo: 06-systemd-service.sh
# Propósito: Configurar servicio systemd para VibeVoice
# Versión: 1.0.0
# Descripción: Crea y configura servicio systemd para inicio automático
# ============================================================================

set -euo pipefail

# Cargar dependencias
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configuracion/stack-vibe.conf"
source "${SCRIPT_DIR}/../librerias/registrador.sh"
source "${SCRIPT_DIR}/../librerias/ayudante.sh"
source "${SCRIPT_DIR}/../librerias/validador.sh"

# ============================================================================
# FUNCIÓN: configurar_servicio_systemd
# ============================================================================
configurar_servicio_systemd() {
    registrar_inicio_modulo "Configuración de Servicio Systemd"
    
    # Crear usuario de servicio
    crear_usuario_servicio
    
    # Configurar permisos de directorios
    configurar_permisos
    
    # Generar archivo de servicio systemd
    generar_servicio_systemd
    
    # Habilitar servicio
    habilitar_servicio
    
    registrar_fin_modulo "Configuración de Servicio Systemd"
    return 0
}

# ============================================================================
# FUNCIÓN: crear_usuario_servicio
# ============================================================================
crear_usuario_servicio() {
    registrar_info "Creando usuario de servicio..."
    
    if crear_usuario_sistema "${VIBE_SERVICE_USER}" "${VIBE_DIR_BASE}"; then
        registrar_exito "Usuario de servicio configurado: ${VIBE_SERVICE_USER}"
        
        # Agregar usuario al grupo docker
        if verificar_comando_existe docker; then
            usermod -aG docker "${VIBE_SERVICE_USER}" 2>/dev/null || true
            registrar_info "Usuario ${VIBE_SERVICE_USER} agregado al grupo docker"
        fi
        
        return 0
    else
        registrar_advertencia "El usuario ${VIBE_SERVICE_USER} ya existe"
        return 0
    fi
}

# ============================================================================
# FUNCIÓN: configurar_permisos
# ============================================================================
configurar_permisos() {
    registrar_info "Configurando permisos de directorios..."
    
    local directorios=(
        "${VIBE_DIR_BASE}"
        "${VIBE_DIR_DATOS}"
        "${VIBE_DIR_LOGS}"
        "${VIBE_DIR_CONFIG}"
        "${VIBE_DIR_BACKUPS}"
        "${VIBE_DIR_TEMP}"
    )
    
    for dir in "${directorios[@]}"; do
        if [[ -d "${dir}" ]]; then
            chown -R "${VIBE_SERVICE_USER}:${VIBE_SERVICE_GROUP}" "${dir}"
            registrar_debug "Permisos configurados: ${dir}"
        else
            crear_directorio "${dir}" "${VIBE_SERVICE_USER}" "755"
        fi
    done
    
    registrar_exito "Permisos configurados correctamente"
    return 0
}

# ============================================================================
# FUNCIÓN: generar_servicio_systemd
# ============================================================================
generar_servicio_systemd() {
    local service_file="/etc/systemd/system/${VIBE_SERVICE_NAME}.service"
    
    registrar_info "Generando archivo de servicio systemd..."
    
    cat > "${service_file}" <<EOF
[Unit]
Description=VibeVoice - Sistema de transcripción y análisis de voz con IA
Documentation=https://github.com/jhoavera/VibeVoice
After=network.target docker.service
Requires=docker.service
PartOf=docker.service

[Service]
Type=forking
User=${VIBE_SERVICE_USER}
Group=${VIBE_SERVICE_GROUP}
WorkingDirectory=${VIBE_DIR_CONFIG}

# Variables de entorno
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EnvironmentFile=${VIBE_DIR_CONFIG}/.env

# Comandos de inicio
ExecStartPre=-/usr/bin/docker-compose -f ${VIBE_DIR_CONFIG}/docker-compose.yml down
ExecStart=/bin/sh -c 'if [ "${VIBE_API_MODE:-docker}" = "host" ]; then echo "VIBE_API_MODE=host: skipping docker-compose up"; exit 0; fi; /usr/bin/docker-compose -f ${VIBE_DIR_CONFIG}/docker-compose.yml up -d --build'
ExecStop=/usr/bin/docker-compose -f ${VIBE_DIR_CONFIG}/docker-compose.yml down

# Reinicio automático
Restart=${VIBE_SERVICE_RESTART}
RestartSec=${VIBE_SERVICE_RESTART_SEC}

# Límites de recursos
LimitNOFILE=65536
LimitNPROC=4096

# Seguridad
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${VIBE_DIR_BASE} ${VIBE_DIR_DATOS} ${VIBE_DIR_LOGS}

# Logs
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${VIBE_SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

    registrar_exito "Archivo de servicio systemd creado: ${service_file}"
    registrar_auditoria "Servicio systemd configurado: ${VIBE_SERVICE_NAME}"
    
    return 0
}

# ============================================================================
# FUNCIÓN: habilitar_servicio
# ============================================================================
habilitar_servicio() {
    registrar_info "Habilitando servicio systemd..."
    
    # Recargar daemon de systemd
    ejecutar_comando "systemctl daemon-reload" "Recargando daemon de systemd"
    
    # Habilitar servicio para inicio automático
    ejecutar_comando "systemctl enable ${VIBE_SERVICE_NAME}" "Habilitando servicio ${VIBE_SERVICE_NAME}"
    
    registrar_exito "Servicio ${VIBE_SERVICE_NAME} habilitado para inicio automático"
    
    # Mostrar estado del servicio
    registrar_info "Estado del servicio:"
    systemctl status "${VIBE_SERVICE_NAME}" --no-pager || true
    
    echo ""
    registrar_info "═══════════════════════════════════════════════════════════════"
    registrar_info "COMANDOS DE GESTIÓN DEL SERVICIO"
    registrar_info "═══════════════════════════════════════════════════════════════"
    registrar_info "Iniciar servicio:    sudo systemctl start ${VIBE_SERVICE_NAME}"
    registrar_info "Detener servicio:    sudo systemctl stop ${VIBE_SERVICE_NAME}"
    registrar_info "Reiniciar servicio:  sudo systemctl restart ${VIBE_SERVICE_NAME}"
    registrar_info "Ver estado:          sudo systemctl status ${VIBE_SERVICE_NAME}"
    registrar_info "Ver logs:            sudo journalctl -u ${VIBE_SERVICE_NAME} -f"
    registrar_info "═══════════════════════════════════════════════════════════════"
    echo ""
    
    return 0
}


# ============================================================================
# FUNCIÓN: generar_servicio_api_host
# Propósito: Crear un servicio systemd que ejecute la API usando el venv del host
# ============================================================================
generar_servicio_api_host() {
    local service_file="/etc/systemd/system/${VIBE_SERVICE_NAME}-api.service"

    registrar_info "Generando servicio systemd para API (host venv): ${service_file}"

    cat > "${service_file}" <<EOF
[Unit]
Description=VibeVoice API (host) - Ejecuta la API con el entorno virtual del host
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=${VIBE_SERVICE_USER}
Group=${VIBE_SERVICE_GROUP}
WorkingDirectory=${VIBE_DIR_BASE}/app
EnvironmentFile=${VIBE_DIR_CONFIG}/.env
ExecStartPre=/bin/sh -c 'if [ -z "${MODEL_PATH}" ]; then echo "MODEL_PATH not set; configure ${VIBE_DIR_CONFIG}/.env and set MODEL_PATH to a HuggingFace repo id or local model path" 1>&2; exit 1; fi'
ExecStart=${VIBE_VENV_DIR}/bin/uvicorn demo.web.app:app --host 0.0.0.0 --port ${VIBE_API_PORT:-8000} --workers 1
Restart=on-failure
RestartSec=5
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=${VIBE_DIR_BASE} ${VIBE_DIR_LOGS}

[Install]
WantedBy=multi-user.target
EOF

    chmod 0644 "${service_file}"
    ejecutar_comando "systemctl daemon-reload" "Recargando daemon de systemd"
    ejecutar_comando "systemctl enable ${VIBE_SERVICE_NAME}-api" "Habilitando servicio ${VIBE_SERVICE_NAME}-api"

    registrar_exito "Servicio ${VIBE_SERVICE_NAME}-api creado y habilitado"
    return 0
}

# ============================================================================
# EJECUCIÓN
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    configurar_servicio_systemd
fi

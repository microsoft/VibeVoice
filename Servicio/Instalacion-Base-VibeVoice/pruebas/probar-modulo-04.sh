#!/bin/bash
# Prueba rápida para crear directorios de datos sin crear usuario
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Variables para prueba (BITACORA necesita estar antes de cargar registrador)
export VIBE_DIR_LOGS="/tmp/vibevoice_test_logs"
export BITACORA="${VIBE_DIR_LOGS}/instalacion-test.log"
mkdir -p "${VIBE_DIR_LOGS}"
chmod 0777 "${VIBE_DIR_LOGS}"

source "${REPO_ROOT}/librerias/ayudante.sh"
source "${REPO_ROOT}/modulos/04-servicios-datos.sh"

# Establecer rutas de prueba después de cargar la configuración para evitar sobrescritura
export VIBE_DIR_BASE="/tmp/vibevoice_test"
export VIBE_DIR_DATOS="${VIBE_DIR_BASE}/datos"
export VIBE_SERVICE_USER="vibevoice"

# Mock crear_usuario_sistema para simular que falla (sin crear usuario real)
crear_usuario_sistema() {
    echo "[MOCK] Simulando fallo al crear usuario: $1" >&2
    return 1
}

# Ejecutar la creación de directorios
echo "Ejecutando crear_directorios_datos (prueba)..."
crear_directorios_datos

echo "Estado de /tmp/vibevoice_test:
"
ls -la "${VIBE_DIR_BASE}" || true
ls -la "${VIBE_DIR_DATOS}" || true

echo "Prueba finalizada"

#!/bin/bash
# ============================================================================
# Script: run-vibevoice.sh
# Propósito: Arranque manual inmediato de VibeVoice en modo desarrollo local
# Versión: 1.0.0
# ============================================================================

set -euo pipefail

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/configuracion/stack-vibe.conf"

# Directorio del proyecto principal
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ============================================================================
# FUNCIÓN: mostrar_ayuda
# ============================================================================
mostrar_ayuda() {
    cat <<EOF
╔══════════════════════════════════════════════════════════════════════╗
║                     EJECUTAR VIBEVOICE (Modo Local)                  ║
╚══════════════════════════════════════════════════════════════════════╝

Uso: $0 [opciones]

Opciones:
  --help              Mostrar esta ayuda
  --port PORT         Puerto para el servidor (default: 8000)
  --host HOST         Host para el servidor (default: 0.0.0.0)
  --workers N         Número de workers (default: 1)
  --reload            Activar hot-reload para desarrollo

Ejemplo:
  $0
  $0 --port 8080 --reload
  $0 --host 127.0.0.1 --port 8000

Variables de entorno (optimización):
  TRANSCRIBE_MODEL    Modelo de transcripción (default: whisper-tiny)
  OMP_NUM_THREADS     Threads OpenMP (default: 1)
  MKL_NUM_THREADS     Threads MKL (default: 1)

EOF
}

# ============================================================================
# PARSEAR ARGUMENTOS
# ============================================================================
PORT="${VIBE_API_PORT:-8000}"
HOST="0.0.0.0"
WORKERS="${UVICORN_WORKERS:-1}"
RELOAD_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            mostrar_ayuda
            exit 0
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --reload)
            RELOAD_FLAG="--reload"
            shift
            ;;
        *)
            echo "Opción desconocida: $1"
            echo "Use --help para ver las opciones disponibles"
            exit 1
            ;;
    esac
done

# ============================================================================
# VERIFICAR ENTORNO VIRTUAL
# ============================================================================
if [[ ! -d "${VIBE_VENV_DIR}" ]]; then
    echo "ERROR: Entorno virtual no encontrado en ${VIBE_VENV_DIR}"
    echo "Ejecuta primero el instalador: sudo ./instalador.sh"
    exit 1
fi

# ============================================================================
# ACTIVAR ENTORNO Y CONFIGURAR VARIABLES
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                   INICIANDO VIBEVOICE (Modo Local)                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Puerto:               ${PORT}"
echo "  Host:                 ${HOST}"
echo "  Workers:              ${WORKERS}"
echo "  Modelo:               ${TRANSCRIBE_MODEL:-whisper-tiny}"
echo "  Entorno virtual:      ${VIBE_VENV_DIR}"
echo "  Directorio proyecto:  ${REPO_ROOT}"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Activar entorno virtual
# shellcheck disable=SC1091
source "${VIBE_VENV_DIR}/bin/activate"

# Configurar variables de optimización
export TRANSCRIBE_MODEL="${TRANSCRIBE_MODEL:-whisper-tiny}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Cambiar al directorio del proyecto
cd "${REPO_ROOT}"

# ============================================================================
# VERIFICAR PUNTO ASGI
# ============================================================================
ASGI_APP="${VIBE_UVICORN_APP:-demo.web.app:app}"

echo "Verificando punto ASGI: ${ASGI_APP}..."
if ! python -c "import sys; mod, obj = '${ASGI_APP}'.split(':'); __import__(mod); getattr(sys.modules[mod], obj)" 2>/dev/null; then
    echo "ADVERTENCIA: No se pudo verificar el punto ASGI ${ASGI_APP}"
    echo "Intentando con puntos alternativos..."
    
    # Probar alternativas comunes
    for candidato in "demo.web.app:app" "app.main:app" "main:app"; do
        echo "  Probando: ${candidato}"
        if python -c "import sys; mod, obj = '${candidato}'.split(':'); __import__(mod); getattr(sys.modules[mod], obj)" 2>/dev/null; then
            ASGI_APP="${candidato}"
            echo "  ✓ Punto ASGI encontrado: ${ASGI_APP}"
            break
        fi
    done
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Iniciando servidor Uvicorn..."
echo "Acceso: http://${HOST}:${PORT}"
echo "Documentación API: http://${HOST}:${PORT}/docs"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# INICIAR SERVIDOR
# ============================================================================
# shellcheck disable=SC2086
python -m uvicorn "${ASGI_APP}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --limit-concurrency "${UVICORN_LIMIT_CONCURRENCY:-1}" \
    ${RELOAD_FLAG}

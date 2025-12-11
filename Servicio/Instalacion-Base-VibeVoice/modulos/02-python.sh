#!/bin/bash
# ============================================================================
# Módulo: 02-python.sh
# Propósito: Instalar y configurar Python para VibeVoice
# Versión: 1.0.0
# Descripción: Instala Python 3.11+, pip, venv y dependencias necesarias
# ============================================================================

set -euo pipefail

# Cargar dependencias
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configuracion/stack-vibe.conf"
source "${SCRIPT_DIR}/../librerias/registrador.sh"
source "${SCRIPT_DIR}/../librerias/ayudante.sh"
source "${SCRIPT_DIR}/../librerias/validador.sh"

# ============================================================================
# FUNCIÓN: instalar_python
# ============================================================================
instalar_python() {
    registrar_inicio_modulo "Instalación de Python ${VIBE_PYTHON_VERSION}"
    
    # Actualizar repositorios
    actualizar_apt || registrar_error_fatal "Error al actualizar repositorios apt"
    
    # Instalar dependencias de Python
    registrar_info "Instalando dependencias de Python..."
    local dependencias=(
        "software-properties-common"
        "build-essential"
        "libssl-dev"
        "libffi-dev"
        "python3-dev"
        "python3-pip"
        "python3-venv"
        "python3-setuptools"
        "python3-wheel"
    )
    
    for paquete in "${dependencias[@]}"; do
        instalar_paquete_apt "${paquete}" || {
            registrar_error "Error al instalar ${paquete}"
            return 1
        }
    done
    
    # Verificar versión de Python
    if validar_python_version; then
        registrar_exito "Python ya está instalado en versión adecuada"
    else
        # Intentar instalar versión específica desde deadsnakes PPA
        registrar_info "Agregando repositorio deadsnakes para Python ${VIBE_PYTHON_VERSION}..."
        
        if ! grep -q "deadsnakes/ppa" /etc/apt/sources.list.d/*.list 2>/dev/null; then
            ejecutar_comando \
                "add-apt-repository -y ppa:deadsnakes/ppa" \
                "Agregando repositorio PPA deadsnakes"
            
            actualizar_apt
        fi
        
        local python_pkg="python${VIBE_PYTHON_VERSION}"
        instalar_paquete_apt "${python_pkg}" || {
            registrar_advertencia "No se pudo instalar ${python_pkg}. Usando versión del sistema."
        }
        
        instalar_paquete_apt "${python_pkg}-dev" 2>/dev/null || true
        instalar_paquete_apt "${python_pkg}-venv" 2>/dev/null || true
    fi
    
    # Crear enlace simbólico si no existe
    if [[ ! -L /usr/bin/python ]]; then
        registrar_info "Creando enlace simbólico para python..."
        ln -sf /usr/bin/python3 /usr/bin/python
    fi
    
    # Actualizar pip
    registrar_info "Actualizando pip..."
    python3 -m pip install --upgrade pip setuptools wheel --quiet || {
        registrar_advertencia "No se pudo actualizar pip"
    }
    
    # Instalar herramientas de Python
    registrar_info "Instalando herramientas de Python..."
    local herramientas=("virtualenv" "pipenv")
    
    for herramienta in "${herramientas[@]}"; do
        python3 -m pip install "${herramienta}" --quiet || {
            registrar_advertencia "No se pudo instalar ${herramienta}"
        }
    done
    
    # Crear entorno virtual para VibeVoice
    registrar_info "Creando entorno virtual en ${VIBE_VENV_DIR}..."
    crear_directorio "$(dirname "${VIBE_VENV_DIR}")"
    
    if [[ ! -d "${VIBE_VENV_DIR}" ]]; then
        python3 -m venv "${VIBE_VENV_DIR}" || {
            registrar_error_fatal "Error al crear entorno virtual"
        }
        registrar_exito "Entorno virtual creado exitosamente"
    else
        registrar_info "Entorno virtual ya existe"
    fi
    
    # Activar entorno virtual e instalar paquetes base
    registrar_info "Instalando paquetes Python en entorno virtual..."
    source "${VIBE_VENV_DIR}/bin/activate"
    
    # Actualizar pip en el venv
    pip install --upgrade pip setuptools wheel --quiet
    
    # Instalar paquetes esenciales
    local paquetes_esenciales=(
        "numpy"
        "scipy"
        "pandas"
        "requests"
        "pyyaml"
        "python-dotenv"
        "psycopg2-binary"
        "redis"
        "pydantic"
        "fastapi"
        "uvicorn[standard]"
    )
    
    for paquete in "${paquetes_esenciales[@]}"; do
        registrar_info "Instalando ${paquete}..."
        pip install "${paquete}" --quiet || {
            registrar_advertencia "Error al instalar ${paquete}"
        }
    done
    
    # Instalar dependencias de VibeVoice si existe requirements.txt
    if [[ -f "${VIBE_DIR_BASE}/../vibevoice/requirements.txt" ]]; then
        registrar_info "Instalando dependencias desde requirements.txt..."
        pip install -r "${VIBE_DIR_BASE}/../vibevoice/requirements.txt" --quiet || {
            registrar_advertencia "Error al instalar dependencias desde requirements.txt"
        }
    fi
    
    deactivate
    
    # Verificar instalación
    registrar_info "Verificando instalación de Python..."
    python3 --version
    pip3 --version
    
    # Mostrar resumen
    echo ""
    registrar_info "═══════════════════════════════════════════════════════════════"
    registrar_info "RESUMEN DE INSTALACIÓN DE PYTHON"
    registrar_info "═══════════════════════════════════════════════════════════════"
    registrar_info "Versión de Python: $(python3 --version | awk '{print $2}')"
    registrar_info "Versión de pip: $(pip3 --version | awk '{print $2}')"
    registrar_info "Entorno virtual: ${VIBE_VENV_DIR}"
    registrar_info "═══════════════════════════════════════════════════════════════"
    echo ""
    
    registrar_fin_modulo "Instalación de Python"
    return 0
}

# ============================================================================
# EJECUCIÓN
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    instalar_python
fi

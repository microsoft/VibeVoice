#!/bin/bash
# ============================================================================
# Archivo: ayudante.sh
# Propósito: Funciones auxiliares y utilidades comunes
# Versión: 1.0.0
# Descripción: Proporciona funciones de ayuda para operaciones comunes
# ============================================================================

# Cargar dependencias
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configuracion/stack-vibe.conf" 2>/dev/null || true
source "${SCRIPT_DIR}/registrador.sh" 2>/dev/null || true

# ============================================================================
# FUNCIÓN: verificar_root
# Propósito: Verificar si el script se ejecuta con privilegios de root
# Retorno: 0 si es root, 1 si no lo es
# ============================================================================
verificar_root() {
    if [[ $EUID -ne 0 ]]; then
        registrar_error "Este script debe ejecutarse con privilegios de root (sudo)"
        return 1
    fi
    registrar_debug "Verificación de privilegios root: OK"
    return 0
}

# ============================================================================
# FUNCIÓN: solicitar_confirmacion
# Propósito: Solicitar confirmación del usuario antes de continuar
# Parámetros:
#   $1: Mensaje de confirmación
# Retorno: 0 si confirma (o si NONINTERACTIVE=1), 1 si rechaza
# ============================================================================
solicitar_confirmacion() {
    local mensaje="${1:-¿Desea continuar?}"
    local respuesta
    
    # En modo NONINTERACTIVE, asumir 'sí' automáticamente
    if [[ "${NONINTERACTIVE:-0}" == "1" ]]; then
        registrar_info "Modo NONINTERACTIVE: auto-confirmando: ${mensaje}"
        return 0
    fi
    
    echo ""
    echo -e "${COLOR_AMARILLO}${mensaje} (s/N)${COLOR_RESET}"
    read -r respuesta
    
    if [[ "${respuesta}" =~ ^[Ss]$ ]]; then
        registrar_auditoria "Usuario confirmó: ${mensaje}"
        return 0
    else
        registrar_advertencia "Usuario rechazó: ${mensaje}"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: ejecutar_comando
# Propósito: Ejecutar comando con registro automático
# Parámetros:
#   $1: Comando a ejecutar
#   $2: Mensaje descriptivo (opcional)
# Retorno: Código de salida del comando
# ============================================================================
ejecutar_comando() {
    local comando="$1"
    local descripcion="${2:-Ejecutando comando}"
    
    registrar_info "${descripcion}..."
    registrar_debug "Comando: ${comando}"
    
    local salida
    salida=$(eval "${comando}" 2>&1)
    local codigo=$?
    
    registrar_comando "${comando}" "${codigo}" "${salida}"
    
    if [[ ${codigo} -ne 0 ]]; then
        registrar_error "Falló: ${descripcion}"
        return ${codigo}
    fi
    
    registrar_exito "${descripcion} completado"
    return 0
}

# ============================================================================
# FUNCIÓN: verificar_comando_existe
# Propósito: Verificar si un comando está disponible en el sistema
# Parámetros:
#   $1: Nombre del comando
# Retorno: 0 si existe, 1 si no existe
# ============================================================================
verificar_comando_existe() {
    local comando="$1"
    
    if command -v "${comando}" &> /dev/null; then
        registrar_debug "Comando encontrado: ${comando}"
        return 0
    else
        registrar_advertencia "Comando no encontrado: ${comando}"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: instalar_paquete_apt
# Propósito: Instalar paquete usando apt con verificación
# Parámetros:
#   $1: Nombre del paquete
# Retorno: 0 si se instaló correctamente, 1 si falló
# ============================================================================
instalar_paquete_apt() {
    local paquete="$1"
    
    registrar_info "Instalando paquete: ${paquete}"
    
    if dpkg -l | grep -q "^ii  ${paquete} "; then
        registrar_info "Paquete ${paquete} ya está instalado"
        return 0
    fi
    
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${paquete}" &>/dev/null
    local codigo=$?
    
    if [[ ${codigo} -eq 0 ]]; then
        registrar_exito "Paquete instalado: ${paquete}"
        return 0
    else
        registrar_error "Error al instalar paquete: ${paquete}"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: actualizar_apt
# Propósito: Actualizar índice de paquetes apt
# Retorno: 0 si se actualizó correctamente, 1 si falló
# ============================================================================
actualizar_apt() {
    registrar_info "Actualizando índice de paquetes apt..."
    
    apt-get update -qq &>/dev/null
    local codigo=$?
    
    if [[ ${codigo} -eq 0 ]]; then
        registrar_exito "Índice de paquetes actualizado"
        return 0
    else
        registrar_error "Error al actualizar índice de paquetes"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: crear_directorio
# Propósito: Crear directorio con permisos y propietario
# Parámetros:
#   $1: Ruta del directorio
#   $2: Propietario (opcional)
#   $3: Permisos (opcional, por defecto 755)
# Retorno: 0 si se creó correctamente, 1 si falló
# ============================================================================
crear_directorio() {
    local ruta="$1"
    local propietario="${2:-}"
    local permisos="${3:-755}"
    
    if [[ -d "${ruta}" ]]; then
        registrar_debug "Directorio ya existe: ${ruta}"
        return 0
    fi
    
    registrar_info "Creando directorio: ${ruta}"
    mkdir -p "${ruta}"
    chmod "${permisos}" "${ruta}"
    
    if [[ -n "${propietario}" ]]; then
        # Solo ejecutar chown si el usuario (y grupo) existen para evitar fallos
        if id "${propietario}" &>/dev/null; then
            chown -R "${propietario}:${propietario}" "${ruta}"
        else
            registrar_advertencia "Propietario especificado no existe: ${propietario} - omitiendo chown"
        fi
    fi
    
    registrar_exito "Directorio creado: ${ruta}"
    return 0
}

# ============================================================================
# FUNCIÓN: crear_usuario_sistema
# Propósito: Crear usuario de sistema para el servicio
# Parámetros:
#   $1: Nombre de usuario
#   $2: Directorio home (opcional)
# Retorno: 0 si se creó correctamente, 1 si falló
# ============================================================================
crear_usuario_sistema() {
    local usuario="$1"
    local home_dir="${2:-/opt/${usuario}}"
    
    if id "${usuario}" &>/dev/null; then
        registrar_info "Usuario ${usuario} ya existe"
        return 0
    fi
    
    registrar_info "Creando usuario de sistema: ${usuario}"
    useradd --system --home-dir "${home_dir}" --shell /bin/bash --create-home "${usuario}"
    local codigo=$?
    
    if [[ ${codigo} -eq 0 ]]; then
        registrar_exito "Usuario creado: ${usuario}"
        registrar_auditoria "Usuario de sistema creado: ${usuario} (home: ${home_dir})"
        return 0
    else
        registrar_error "Error al crear usuario: ${usuario}"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: obtener_memoria_total_gb
# Propósito: Obtener memoria RAM total del sistema en GB
# Retorno: Memoria total en GB
# ============================================================================
obtener_memoria_total_gb() {
    local mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local mem_gb=$((mem_kb / 1024 / 1024))
    echo "${mem_gb}"
}

# ============================================================================
# FUNCIÓN: obtener_espacio_disco_gb
# Propósito: Obtener espacio disponible en disco en GB
# Parámetros:
#   $1: Ruta del punto de montaje (por defecto /)
# Retorno: Espacio disponible en GB
# ============================================================================
obtener_espacio_disco_gb() {
    local ruta="${1:-/}"
    local espacio=$(df -BG "${ruta}" | tail -1 | awk '{print $4}' | sed 's/G//')
    echo "${espacio}"
}

# ============================================================================
# FUNCIÓN: obtener_numero_cpus
# Propósito: Obtener número de CPUs del sistema
# Retorno: Número de CPUs
# ============================================================================
obtener_numero_cpus() {
    nproc
}

# ============================================================================
# FUNCIÓN: verificar_conectividad_internet
# Propósito: Verificar si hay conexión a Internet
# Retorno: 0 si hay conexión, 1 si no la hay
# ============================================================================
verificar_conectividad_internet() {
    registrar_debug "Verificando conectividad a Internet..."
    
    if ping -c 1 -W 2 8.8.8.8 &>/dev/null; then
        registrar_debug "Conectividad a Internet: OK"
        return 0
    else
        registrar_advertencia "No se detectó conexión a Internet"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: esperar_servicio
# Propósito: Esperar a que un servicio esté activo
# Parámetros:
#   $1: Nombre del servicio
#   $2: Timeout en segundos (opcional, por defecto 30)
# Retorno: 0 si el servicio está activo, 1 si timeout
# ============================================================================
esperar_servicio() {
    local servicio="$1"
    local timeout="${2:-30}"
    local contador=0
    
    registrar_info "Esperando a que el servicio ${servicio} esté activo..."
    
    while [[ ${contador} -lt ${timeout} ]]; do
        if systemctl is-active --quiet "${servicio}"; then
            registrar_exito "Servicio ${servicio} está activo"
            return 0
        fi
        sleep 1
        ((contador++))
    done
    
    registrar_error "Timeout esperando al servicio ${servicio}"
    return 1
}

# ============================================================================
# FUNCIÓN: wait_for_postgres
# Propósito: Esperar a que PostgreSQL esté listo para aceptar conexiones
# Parámetros:
#   $1: Nombre del contenedor (opcional, por defecto vibevoice-postgres)
#   $2: Timeout en segundos (opcional, por defecto 60)
# Retorno: 0 si PostgreSQL está listo, 1 si timeout
# ============================================================================
wait_for_postgres() {
    local container_name="${1:-vibevoice-postgres}"
    local timeout="${2:-60}"
    local contador=0
    
    registrar_info "Esperando a que PostgreSQL (${container_name}) esté listo..."
    
    while [[ ${contador} -lt ${timeout} ]]; do
        if docker exec "${container_name}" pg_isready -U "${VIBE_POSTGRES_USER:-postgres}" &>/dev/null; then
            registrar_exito "PostgreSQL está listo y aceptando conexiones"
            return 0
        fi
        sleep 2
        ((contador+=2))
    done
    
    registrar_error "Timeout esperando a PostgreSQL (${timeout}s). El contenedor puede no estar funcionando correctamente."
    registrar_error "Revisa los logs con: docker logs ${container_name}"
    return 1
}

# ============================================================================
# FUNCIÓN: wait_for_redis
# Propósito: Esperar a que Redis esté listo para aceptar conexiones
# Parámetros:
#   $1: Nombre del contenedor (opcional, por defecto vibevoice-redis)
#   $2: Timeout en segundos (opcional, por defecto 30)
# Retorno: 0 si Redis está listo, 1 si timeout
# ============================================================================
wait_for_redis() {
    local container_name="${1:-vibevoice-redis}"
    local timeout="${2:-30}"
    local contador=0
    
    registrar_info "Esperando a que Redis (${container_name}) esté listo..."
    
    while [[ ${contador} -lt ${timeout} ]]; do
        if docker exec "${container_name}" redis-cli -a "${VIBE_REDIS_PASSWORD:-}" ping 2>/dev/null | grep -q "PONG"; then
            registrar_exito "Redis está listo y respondiendo a PING"
            return 0
        fi
        sleep 1
        ((contador++))
    done
    
    registrar_error "Timeout esperando a Redis (${timeout}s). El contenedor puede no estar funcionando correctamente."
    registrar_error "Revisa los logs con: docker logs ${container_name}"
    return 1
}

# ============================================================================
# FUNCIÓN: comparar_versiones
# Propósito: Comparar dos versiones en formato semántico
# Parámetros:
#   $1: Versión 1
#   $2: Versión 2
# Retorno: 0 si v1 >= v2, 1 si v1 < v2
# ============================================================================
comparar_versiones() {
    local v1="$1"
    local v2="$2"
    
    if [[ "${v1}" == "${v2}" ]]; then
        return 0
    fi
    
    local IFS=.
    local i ver1=($v1) ver2=($v2)
    
    for ((i=0; i<${#ver1[@]} || i<${#ver2[@]}; i++)); do
        local num1=${ver1[i]:-0}
        local num2=${ver2[i]:-0}
        
        if ((10#$num1 > 10#$num2)); then
            return 0
        elif ((10#$num1 < 10#$num2)); then
            return 1
        fi
    done
    
    return 0
}

# ============================================================================
# FUNCIÓN: generar_password_seguro
# Propósito: Generar contraseña aleatoria segura
# Parámetros:
#   $1: Longitud (opcional, por defecto 32)
# Retorno: Contraseña generada
# ============================================================================
generar_password_seguro() {
    local longitud="${1:-32}"
    openssl rand -base64 48 | tr -d "=+/" | cut -c1-${longitud}
}

# ============================================================================
# FUNCIÓN: crear_respaldo
# Propósito: Crear respaldo de un archivo o directorio
# Parámetros:
#   $1: Ruta del archivo/directorio a respaldar
# Retorno: 0 si se creó el respaldo, 1 si falló
# ============================================================================
crear_respaldo() {
    local origen="$1"
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local respaldo="${origen}.backup-${timestamp}"
    
    if [[ ! -e "${origen}" ]]; then
        registrar_advertencia "No existe el archivo/directorio a respaldar: ${origen}"
        return 1
    fi
    
    registrar_info "Creando respaldo: ${respaldo}"
    cp -r "${origen}" "${respaldo}"
    
    if [[ $? -eq 0 ]]; then
        registrar_exito "Respaldo creado: ${respaldo}"
        return 0
    else
        registrar_error "Error al crear respaldo de: ${origen}"
        return 1
    fi
}

# ============================================================================
# FUNCIÓN: mostrar_barra_progreso
# Propósito: Mostrar barra de progreso simple
# Parámetros:
#   $1: Progreso actual (0-100)
#   $2: Mensaje (opcional)
# ============================================================================
mostrar_barra_progreso() {
    local progreso="$1"
    local mensaje="${2:-Procesando}"
    local ancho=50
    local completado=$((progreso * ancho / 100))
    local restante=$((ancho - completado))
    
    printf "\r%s [" "${mensaje}"
    printf "%${completado}s" | tr ' ' '='
    printf "%${restante}s" | tr ' ' '-'
    printf "] %3d%%" "${progreso}"
    
    if [[ ${progreso} -eq 100 ]]; then
        echo ""
    fi
}

# ============================================================================
# EXPORTAR FUNCIONES
# ============================================================================
export -f verificar_root
export -f solicitar_confirmacion
export -f ejecutar_comando
export -f verificar_comando_existe
export -f instalar_paquete_apt
export -f actualizar_apt
export -f crear_directorio
export -f crear_usuario_sistema
export -f obtener_memoria_total_gb
export -f obtener_espacio_disco_gb
export -f obtener_numero_cpus
export -f verificar_conectividad_internet
export -f esperar_servicio
export -f wait_for_postgres
export -f wait_for_redis
export -f comparar_versiones
export -f generar_password_seguro
export -f crear_respaldo
export -f mostrar_barra_progreso

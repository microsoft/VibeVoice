# LEEME - Servicio de Instalación VibeVoice

## Descripción General

Este es el **Servicio de Instalación Base para VibeVoice**, un sistema completo, modular e idempotente diseñado para preparar y desplegar VibeVoice en sistemas Ubuntu 22.04 y 24.04.

## Características Principales

- ✅ **Modular**: Instalación dividida en 6 módulos independientes
- ✅ **Idempotente**: Puede ejecutarse múltiples veces sin efectos adversos
- ✅ **Trazabilidad**: Sistema completo de bitácoras y auditoría
- ✅ **Validación**: Pruebas automáticas pre y post-instalación
- ✅ **Documentación**: Documentación técnica en español empresarial
- ✅ **Automatización**: Servicio systemd para gestión del ciclo de vida

## Requisitos del Sistema

### Sistema Operativo
- Ubuntu 22.04 LTS (Jammy Jellyfish)
- Ubuntu 24.04 LTS (Noble Numbat)

### Recursos Mínimos
- **RAM**: 8 GB (recomendado: 16 GB)
- **CPU**: 4 núcleos (recomendado: 8 núcleos)
- **Disco**: 50 GB de espacio libre (recomendado: 100 GB)
- **Red**: Conexión a Internet estable

### Privilegios
- Acceso root o sudo

## Estructura del Proyecto

```
Servicio/Instalacion-Base-VibeVoice/
├── instalador.sh                    # Script principal de instalación
├── configuracion/
│   └── stack-vibe.conf             # Única fuente de verdad (configuración)
├── librerias/
│   ├── registrador.sh              # Sistema de bitácoras
│   ├── ayudante.sh                 # Funciones auxiliares
│   └── validador.sh                # Validaciones del sistema
├── modulos/
│   ├── 01-verificacion-sistema.sh  # Verificación de requisitos
│   ├── 02-python.sh                # Instalación de Python
│   ├── 03-docker.sh                # Instalación de Docker
│   ├── 04-servicios-datos.sh       # PostgreSQL, Redis, Kafka
│   ├── 05-docker-compose.sh        # Orquestación de contenedores
│   └── 06-systemd-service.sh       # Servicio de sistema
├── pruebas/
│   ├── validar-instalacion.sh      # Validación post-instalación
│   └── probar-modulo.sh            # Pruebas de módulos individuales
├── bitacoras/                       # Logs de instalación (generado)
├── LEEME.md                         # Este archivo
├── GLOSARIO.md                      # Glosario de términos técnicos
└── arbol.md                         # Estructura detallada del proyecto
```

## Instalación Rápida

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/jhoavera/VibeVoice.git
cd VibeVoice/Servicio/Instalacion-Base-VibeVoice
```

### Paso 2: Hacer Ejecutables los Scripts

```bash
chmod +x instalador.sh
chmod +x librerias/*.sh
chmod +x modulos/*.sh
chmod +x pruebas/*.sh
```

### Paso 3: Verificar Requisitos (Opcional)

```bash
sudo ./instalador.sh --check
```

### Paso 4: Ejecutar Instalación

```bash
sudo ./instalador.sh
```

La instalación tardará aproximadamente 15-30 minutos dependiendo de la velocidad de su conexión a Internet y los recursos del sistema.

## Componentes Instalados

### 1. Python (Módulo 02)
- Python 3.11 o superior
- pip, virtualenv, pipenv
- Entorno virtual en `/opt/vibevoice/venv`
- Paquetes: numpy, scipy, pandas, fastapi, uvicorn, psycopg2, redis, pydantic

### 2. Docker (Módulo 03)
- Docker Engine 24.0+
- Docker Compose 2.20+
- Red Docker: `vibevoice-network`

### 3. Servicios de Datos (Módulo 04)

#### PostgreSQL
- Versión: 15-alpine
- Puerto: 5432
- Base de datos: `vibevoice_db`
- Usuario: `vibe_admin`
- **⚠️ ADVERTENCIA**: Cambiar contraseña por defecto en producción

#### Redis
- Versión: 7-alpine
- Puerto: 6379
- Política de memoria: allkeys-lru
- Persistencia: AOF habilitado

#### Kafka + Zookeeper
- Kafka 3.5 (puerto 9092)
- Zookeeper (puerto 2181)
- Topics: audio-input, transcription-output, analysis-results

### 4. Docker Compose (Módulo 05)
- Archivo de orquestación: `/opt/vibevoice/config/docker-compose.yml`
- Variables de entorno: `/opt/vibevoice/config/.env`

### 5. Servicio Systemd (Módulo 06)
- Servicio: `vibevoice.service`
- Usuario del sistema: `vibevoice`
- Inicio automático configurado

## Uso Post-Instalación

### Iniciar Servicios

```bash
sudo systemctl start vibevoice
```

### Detener Servicios

```bash
sudo systemctl stop vibevoice
```

### Reiniciar Servicios

```bash
sudo systemctl restart vibevoice
```

### Ver Estado

```bash
sudo systemctl status vibevoice
```

### Ver Logs en Tiempo Real

```bash
sudo journalctl -u vibevoice -f
```

### Validar Instalación

```bash
cd /path/to/Servicio/Instalacion-Base-VibeVoice
sudo ./pruebas/validar-instalacion.sh
```

## Configuración

### Archivo Principal de Configuración

El archivo `configuracion/stack-vibe.conf` es la **única fuente de verdad** para todos los parámetros de instalación. Edite este archivo antes de la instalación para personalizar:

- Rutas de directorios
- Versiones de software
- Puertos de servicios
- Credenciales (⚠️ cambiar en producción)
- Recursos del sistema
- Políticas de logs y backups

### Variables de Entorno

Después de la instalación, el archivo `/opt/vibevoice/config/.env` contiene las variables de entorno para Docker Compose. Este archivo contiene credenciales sensibles y debe protegerse adecuadamente.

## Pruebas y Validación

### Validación Completa

```bash
sudo ./pruebas/validar-instalacion.sh
```

Ejecuta una batería completa de pruebas que verifican:
- Comandos del sistema instalados
- Versiones de software correctas
- Directorios creados
- Servicios Docker funcionando
- Archivos de configuración presentes
- Servicio systemd configurado
- Puertos en uso

### Prueba de Módulos Individuales

```bash
# Listar módulos disponibles
./pruebas/probar-modulo.sh --list

# Probar un módulo específico
sudo ./pruebas/probar-modulo.sh 02  # Ejemplo: módulo de Python
```

## Bitácoras y Auditoría

### Ubicación de Logs

- **Logs de instalación**: `/opt/vibevoice/logs/` o `./bitacoras/`
- **Log principal**: `instalacion-YYYYMMDD-HHMMSS.log`
- **Log de errores**: `errores-YYYYMMDD-HHMMSS.log`
- **Log de auditoría**: `auditoria-YYYYMMDD-HHMMSS.log`

### Niveles de Log

- **INFO**: Información general del proceso
- **SUCCESS**: Operaciones completadas exitosamente
- **WARN**: Advertencias que no impiden la instalación
- **ERROR**: Errores que pueden afectar la instalación
- **DEBUG**: Información detallada (cuando `VIBE_LOG_LEVEL=DEBUG`)
- **AUDIT**: Registro de auditoría de operaciones críticas

## Seguridad

### ⚠️ Consideraciones Críticas de Seguridad

1. **Cambiar Contraseñas por Defecto**
   - PostgreSQL: `VIBE_POSTGRES_PASSWORD`
   - Redis: `VIBE_REDIS_PASSWORD`
   - Secret Key: `VIBE_SECRET_KEY`

2. **Proteger Archivo .env**
   ```bash
   chmod 600 /opt/vibevoice/config/.env
   ```

3. **Firewall**
   - Configurar UFW o iptables para limitar acceso a puertos
   - Solo exponer puertos necesarios

4. **SSL/TLS**
   - Habilitar SSL para PostgreSQL
   - Configurar HTTPS para API
   - Usar certificados válidos en producción

5. **Secret Manager**
   - Usar AWS Secrets Manager, HashiCorp Vault, o similar en producción
   - No almacenar credenciales en archivos de configuración

6. **Backups**
   - Configurar backups automáticos de bases de datos
   - Almacenar backups en ubicación segura y cifrada

## Solución de Problemas

### Error: "Sistema operativo no soportado"
**Solución**: Verificar que esté usando Ubuntu 22.04 o 24.04
```bash
lsb_release -a
```

### Error: "Recursos del sistema insuficientes"
**Solución**: Verificar RAM, CPU y disco disponible
```bash
free -h
nproc
df -h
```

### Error: "Puerto ya está en uso"
**Solución**: Identificar y detener el servicio que usa el puerto
```bash
sudo netstat -tulpn | grep :5432
sudo systemctl stop <servicio-conflictivo>
```

### Error: "Docker no funciona correctamente"
**Solución**: Reiniciar servicio Docker
```bash
sudo systemctl restart docker
sudo systemctl status docker
```

### Error: "Permisos insuficientes"
**Solución**: Ejecutar con sudo
```bash
sudo ./instalador.sh
```

## Desinstalación

Para desinstalar VibeVoice:

```bash
# Detener servicios
sudo systemctl stop vibevoice
sudo systemctl disable vibevoice

# Eliminar contenedores Docker
docker-compose -f /opt/vibevoice/config/docker-compose.yml down -v

# Eliminar servicio systemd
sudo rm /etc/systemd/system/vibevoice.service
sudo systemctl daemon-reload

# Eliminar directorios (ADVERTENCIA: elimina datos)
sudo rm -rf /opt/vibevoice

# Eliminar usuario de servicio
sudo userdel -r vibevoice
```

## Opciones de Línea de Comandos

```bash
# Mostrar ayuda
./instalador.sh --help

# Mostrar versión
./instalador.sh --version

# Solo verificar requisitos
sudo ./instalador.sh --check

# Instalación sin confirmación (modo automatizado)
sudo ./instalador.sh --skip-confirmation
```

## Soporte y Contribuciones

### Reportar Problemas
- GitHub Issues: https://github.com/jhoavera/VibeVoice/issues

### Documentación Adicional
- `GLOSARIO.md`: Glosario de términos técnicos
- `arbol.md`: Estructura detallada del proyecto
- Repositorio principal: https://github.com/jhoavera/VibeVoice

### Licencia
Ver archivo `LICENSE` en la raíz del repositorio.

## Contacto

Para más información sobre VibeVoice, visite:
- https://github.com/jhoavera/VibeVoice

---

**Última actualización**: Diciembre 2024  
**Versión del instalador**: 1.0.0  
**Autor**: jhoavera

# Estructura del Proyecto - Servicio de Instalación VibeVoice

## Árbol de Directorios Completo

```
Servicio/Instalacion-Base-VibeVoice/
│
├── instalador.sh                           # Script principal de instalación
│   │
│   ├─> Orquesta la ejecución de todos los módulos
│   ├─> Maneja argumentos de línea de comandos
│   ├─> Gestiona el flujo de instalación
│   └─> Genera reportes y estadísticas finales
│
├── configuracion/
│   └── stack-vibe.conf                     # Única fuente de verdad
│       │
│       ├─> Configuración de rutas del sistema
│       ├─> Versiones de software a instalar
│       ├─> Credenciales de servicios (cambiar en producción)
│       ├─> Parámetros de recursos y límites
│       └─> Configuración de logs y backups
│
├── librerias/
│   │
│   ├── registrador.sh                      # Sistema de bitácoras
│   │   │
│   │   ├─> inicializar_bitacoras(): Configuración inicial
│   │   ├─> registrar_info(): Mensajes informativos
│   │   ├─> registrar_exito(): Operaciones exitosas
│   │   ├─> registrar_advertencia(): Advertencias
│   │   ├─> registrar_error(): Errores
│   │   ├─> registrar_debug(): Información de depuración
│   │   ├─> registrar_auditoria(): Registro de auditoría
│   │   ├─> registrar_comando(): Log de comandos ejecutados
│   │   ├─> registrar_error_fatal(): Errores críticos (termina)
│   │   └─> obtener_estadisticas_bitacoras(): Resumen final
│   │
│   ├── ayudante.sh                         # Funciones auxiliares
│   │   │
│   │   ├─> verificar_root(): Validar privilegios
│   │   ├─> solicitar_confirmacion(): Interacción con usuario
│   │   ├─> ejecutar_comando(): Ejecución con logging
│   │   ├─> verificar_comando_existe(): Verificar disponibilidad
│   │   ├─> instalar_paquete_apt(): Instalación de paquetes
│   │   ├─> actualizar_apt(): Actualizar repositorios
│   │   ├─> crear_directorio(): Crear con permisos
│   │   ├─> crear_usuario_sistema(): Crear usuario de servicio
│   │   ├─> obtener_memoria_total_gb(): Info de RAM
│   │   ├─> obtener_espacio_disco_gb(): Info de disco
│   │   ├─> obtener_numero_cpus(): Info de CPUs
│   │   ├─> verificar_conectividad_internet(): Test de red
│   │   ├─> esperar_servicio(): Esperar servicio activo
│   │   ├─> comparar_versiones(): Comparar versiones semánticas
│   │   ├─> generar_password_seguro(): Generar contraseñas
│   │   ├─> crear_respaldo(): Crear backups
│   │   └─> mostrar_barra_progreso(): UI de progreso
│   │
│   └── validador.sh                        # Validaciones del sistema
│       │
│       ├─> validar_ubuntu_version(): Verificar SO compatible
│       ├─> validar_recursos_sistema(): RAM, CPU, disco
│       ├─> validar_conectividad(): Internet y repos
│       ├─> validar_python_version(): Versión de Python
│       ├─> validar_docker_version(): Versión de Docker
│       ├─> validar_docker_compose_version(): Versión de Compose
│       ├─> validar_puertos_disponibles(): Puertos libres
│       ├─> validar_privilegios(): Permisos root
│       ├─> validar_configuracion(): Archivo de config
│       ├─> validar_servicio_activo(): Estado de servicio
│       ├─> validar_docker_funcionando(): Docker operativo
│       └─> validar_instalacion_completa(): Validación total
│
├── modulos/
│   │
│   ├── 01-verificacion-sistema.sh          # Módulo 1: Verificaciones
│   │   │
│   │   ├─> Validar privilegios de root
│   │   ├─> Validar versión de Ubuntu (22.04/24.04)
│   │   ├─> Verificar recursos mínimos (RAM, CPU, disco)
│   │   ├─> Verificar conectividad a Internet
│   │   ├─> Validar archivo de configuración
│   │   ├─> Verificar puertos disponibles
│   │   └─> Mostrar resumen del sistema
│   │
│   ├── 02-python.sh                        # Módulo 2: Python
│   │   │
│   │   ├─> Actualizar repositorios apt
│   │   ├─> Instalar dependencias de compilación
│   │   ├─> Agregar PPA deadsnakes (si necesario)
│   │   ├─> Instalar Python 3.11+
│   │   ├─> Instalar pip, virtualenv, pipenv
│   │   ├─> Crear entorno virtual en /opt/vibevoice/venv
│   │   ├─> Instalar paquetes esenciales en venv
│   │   └─> Validar instalación de Python
│   │
│   ├── 03-docker.sh                        # Módulo 3: Docker
│   │   │
│   │   ├─> Verificar instalación previa
│   │   ├─> Instalar dependencias previas
│   │   ├─> Agregar clave GPG de Docker
│   │   ├─> Configurar repositorio de Docker
│   │   ├─> Instalar Docker Engine y plugins
│   │   ├─> Instalar Docker Compose standalone
│   │   ├─> Iniciar y habilitar servicio Docker
│   │   ├─> Configurar daemon.json
│   │   ├─> Crear red vibevoice-network
│   │   ├─> Agregar usuarios al grupo docker
│   │   └─> Validar funcionamiento de Docker
│   │
│   ├── 04-servicios-datos.sh               # Módulo 4: Servicios de Datos
│   │   │
│   │   ├─> Crear directorios de datos
│   │   │
│   │   ├─> PostgreSQL:
│   │   │   ├─> Detener contenedor existente
│   │   │   ├─> Crear contenedor PostgreSQL 15-alpine
│   │   │   ├─> Configurar variables de entorno
│   │   │   ├─> Montar volumen de datos
│   │   │   ├─> Exponer puerto 5432
│   │   │   ├─> Esperar disponibilidad
│   │   │   └─> Crear extensiones (uuid-ossp, pg_trgm)
│   │   │
│   │   ├─> Redis:
│   │   │   ├─> Detener contenedor existente
│   │   │   ├─> Crear contenedor Redis 7-alpine
│   │   │   ├─> Configurar autenticación y memoria
│   │   │   ├─> Habilitar persistencia AOF
│   │   │   ├─> Exponer puerto 6379
│   │   │   └─> Validar con PING
│   │   │
│   │   └─> Kafka + Zookeeper:
│   │       ├─> Crear contenedor Zookeeper
│   │       ├─> Exponer puerto 2181
│   │       ├─> Crear contenedor Kafka
│   │       ├─> Configurar conexión a Zookeeper
│   │       ├─> Exponer puerto 9092
│   │       └─> Crear topics predefinidos
│   │
│   ├── 05-docker-compose.sh                # Módulo 5: Docker Compose
│   │   │
│   │   ├─> Crear directorio de configuración
│   │   ├─> Generar docker-compose.yml:
│   │   │   ├─> Definir servicios (postgres, redis, kafka, etc.)
│   │   │   ├─> Configurar redes
│   │   │   ├─> Configurar volúmenes
│   │   │   ├─> Configurar health checks
│   │   │   └─> Configurar políticas de reinicio
│   │   ├─> Generar archivo .env:
│   │   │   ├─> Variables de PostgreSQL
│   │   │   ├─> Variables de Redis
│   │   │   ├─> Variables de Kafka
│   │   │   ├─> Variables de API
│   │   │   ├─> Variables de seguridad
│   │   │   └─> Establecer permisos restrictivos (600)
│   │   └─> Validar sintaxis de docker-compose.yml
│   │
│   └── 06-systemd-service.sh               # Módulo 6: Servicio Systemd
│       │
│       ├─> Crear usuario de sistema 'vibevoice'
│       ├─> Configurar permisos de directorios
│       ├─> Generar archivo vibevoice.service:
│       │   ├─> Configurar dependencias (After, Requires)
│       │   ├─> Configurar usuario y grupo
│       │   ├─> Definir comandos (ExecStart, ExecStop)
│       │   ├─> Configurar reinicio automático
│       │   ├─> Establecer límites de recursos
│       │   └─> Aplicar sandboxing de seguridad
│       ├─> Recargar daemon de systemd
│       ├─> Habilitar servicio para inicio automático
│       └─> Mostrar comandos de gestión
│
├── pruebas/
│   │
│   ├── validar-instalacion.sh              # Validación post-instalación
│   │   │
│   │   ├─> Sección 1: Comandos del Sistema
│   │   │   └─> Verificar python3, pip3, docker, docker-compose
│   │   │
│   │   ├─> Sección 2: Versiones de Software
│   │   │   └─> Mostrar versiones instaladas
│   │   │
│   │   ├─> Sección 3: Directorios
│   │   │   └─> Verificar existencia de directorios clave
│   │   │
│   │   ├─> Sección 4: Servicios Docker
│   │   │   ├─> Verificar servicio Docker activo
│   │   │   ├─> Verificar red Docker
│   │   │   └─> Verificar contenedores (si existen)
│   │   │
│   │   ├─> Sección 5: Archivos de Configuración
│   │   │   └─> Verificar existencia de configs
│   │   │
│   │   ├─> Sección 6: Servicio Systemd
│   │   │   └─> Verificar servicio creado y habilitado
│   │   │
│   │   ├─> Sección 7: Conectividad de Puertos
│   │   │   └─> Verificar puertos en uso
│   │   │
│   │   └─> Resumen Final
│   │       ├─> Contador de pruebas
│   │       ├─> Pruebas exitosas vs fallidas
│   │       └─> Porcentaje de éxito
│   │
│   └── probar-modulo.sh                    # Prueba de módulos individuales
│       │
│       ├─> Listar módulos disponibles
│       ├─> Ejecutar módulo específico
│       ├─> Medir tiempo de ejecución
│       └─> Reportar resultado
│
├── bitacoras/                               # Directorio de logs (generado)
│   │
│   ├── instalacion-YYYYMMDD-HHMMSS.log     # Log principal
│   ├── errores-YYYYMMDD-HHMMSS.log         # Log de errores
│   └── auditoria-YYYYMMDD-HHMMSS.log       # Log de auditoría
│
├── LEEME.md                                 # Documentación principal
│   │
│   ├─> Descripción general
│   ├─> Características principales
│   ├─> Requisitos del sistema
│   ├─> Guía de instalación
│   ├─> Componentes instalados
│   ├─> Uso post-instalación
│   ├─> Configuración
│   ├─> Pruebas y validación
│   ├─> Bitácoras y auditoría
│   ├─> Seguridad
│   ├─> Solución de problemas
│   └─> Soporte y contribuciones
│
├── GLOSARIO.md                              # Glosario técnico
│   │
│   ├─> Términos técnicos (A-Z)
│   ├─> Acrónimos comunes
│   ├─> Convenciones de nomenclatura
│   └─> Referencias técnicas
│
└── arbol.md                                 # Este archivo
    │
    ├─> Estructura completa del proyecto
    ├─> Descripción de cada componente
    ├─> Flujo de ejecución
    └─> Dependencias entre módulos

```

## Directorios Creados en el Sistema (Post-Instalación)

```
/opt/vibevoice/                              # Directorio base
│
├── venv/                                    # Entorno virtual Python
│   ├── bin/                                 # Ejecutables
│   ├── lib/                                 # Bibliotecas Python
│   └── include/                             # Headers
│
├── datos/                                   # Datos de servicios
│   ├── postgresql/                          # Datos PostgreSQL
│   ├── redis/                               # Datos Redis
│   ├── kafka/                               # Datos Kafka
│   └── zookeeper/                           # Datos Zookeeper
│
├── logs/                                    # Bitácoras del sistema
│   ├── instalacion-*.log                    # Logs de instalación
│   ├── errores-*.log                        # Logs de errores
│   └── auditoria-*.log                      # Logs de auditoría
│
├── config/                                  # Configuraciones
│   ├── docker-compose.yml                   # Orquestación de servicios
│   └── .env                                 # Variables de entorno
│
├── backups/                                 # Respaldos automáticos
│   └── (archivos de backup)
│
├── temp/                                    # Archivos temporales
│   └── (archivos temporales)
│
└── app/                                     # Aplicación VibeVoice
    └── (código de la aplicación)
```

## Archivos de Systemd

```
/etc/systemd/system/
└── vibevoice.service                        # Servicio de VibeVoice
```

## Flujo de Ejecución del Instalador

```
┌─────────────────────────────────────────────────────────────────┐
│                    INICIO DE INSTALACIÓN                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Cargar Configuración (stack-vibe.conf)                         │
│  Cargar Librerías (registrador, ayudante, validador)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Inicializar Sistema de Bitácoras                               │
│  Crear archivos de log                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Mostrar Banner y Resumen                                       │
│  Solicitar Confirmación del Usuario                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MÓDULO 01: Verificación del Sistema                            │
│  ├─ Validar privilegios root                                    │
│  ├─ Validar versión Ubuntu                                      │
│  ├─ Validar recursos (RAM, CPU, disco)                          │
│  ├─ Validar conectividad                                        │
│  └─ Validar puertos disponibles                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MÓDULO 02: Instalación de Python                               │
│  ├─ Instalar dependencias                                       │
│  ├─ Instalar Python 3.11+                                       │
│  ├─ Crear entorno virtual                                       │
│  └─ Instalar paquetes Python                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MÓDULO 03: Instalación de Docker                               │
│  ├─ Agregar repositorio Docker                                  │
│  ├─ Instalar Docker Engine                                      │
│  ├─ Instalar Docker Compose                                     │
│  ├─ Configurar daemon                                           │
│  └─ Crear red Docker                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MÓDULO 04: Configuración de Servicios de Datos                 │
│  ├─ Desplegar PostgreSQL                                        │
│  ├─ Desplegar Redis                                             │
│  └─ Desplegar Kafka + Zookeeper                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MÓDULO 05: Configuración de Docker Compose                     │
│  ├─ Generar docker-compose.yml                                  │
│  ├─ Generar archivo .env                                        │
│  └─ Validar sintaxis                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MÓDULO 06: Configuración de Servicio Systemd                   │
│  ├─ Crear usuario de sistema                                    │
│  ├─ Configurar permisos                                         │
│  ├─ Generar archivo de servicio                                 │
│  └─ Habilitar servicio                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Mostrar Resultado Final                                        │
│  Mostrar Instrucciones Post-Instalación                         │
│  Mostrar Estadísticas de Bitácoras                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INSTALACIÓN COMPLETADA                        │
└─────────────────────────────────────────────────────────────────┘
```

## Dependencias Entre Módulos

```
Módulo 01 (Verificación)
    │
    ├──> Módulo 02 (Python)
    │       │
    │       └──> Módulo 03 (Docker)
    │               │
    │               ├──> Módulo 04 (Servicios de Datos)
    │               │       │
    │               │       └──> Módulo 05 (Docker Compose)
    │               │               │
    │               └───────────────┴──> Módulo 06 (Systemd)
```

## Tamaños Aproximados de Componentes

| Componente | Tamaño Aproximado |
|------------|-------------------|
| Scripts de instalación | ~150 KB |
| Python + venv + paquetes | ~2-3 GB |
| Docker Engine | ~500 MB |
| PostgreSQL (imagen) | ~200 MB |
| Redis (imagen) | ~30 MB |
| Kafka + Zookeeper (imágenes) | ~600 MB |
| Datos de servicios | Variable (crece con uso) |
| Logs | Variable (configurable) |
| **Total inicial** | **~4-5 GB** |

---

**Última actualización**: Diciembre 2024  
**Versión del documento**: 1.0.0

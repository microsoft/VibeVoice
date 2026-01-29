@echo off
setlocal EnableDelayedExpansion
set "ROOT=%CD%"
echo ========================================
echo   VibeVoice-Narrator Web GUI
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)

REM Verify Python version >= 3.9
set "PY_VER="
for /f "usebackq delims=" %%v in (`python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"`) do set "PY_VER=%%v"
if not defined PY_VER (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)
if not defined PY_MAJOR (
    echo ERROR: Could not detect Python version
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)
rem Normalize numeric values
set /A PY_MAJOR_NUM=!PY_MAJOR! 2>nul
set /A PY_MINOR_NUM=!PY_MINOR! 2>nul
if !PY_MAJOR_NUM! LSS 3 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)
if !PY_MAJOR_NUM! EQU 3 (
    if !PY_MINOR_NUM! LSS 9 (
        echo ERROR: Installed Python version is too old; please install Python 3.9 or higher
        pause
        exit /b 1
    )
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18 or higher
    pause
    exit /b 1
)

REM Verify Node.js major version >= 18
set "NODE_VER="
for /f "usebackq delims=" %%v in (`node --version`) do set "NODE_VER=%%v"
if not defined NODE_VER (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18 or higher
    pause
    exit /b 1
)
rem Strip leading 'v' if present
set "NODE_VER_NOV=!NODE_VER:v=!"
for /f "tokens=1 delims=." %%a in ("!NODE_VER_NOV!") do set "NODE_MAJOR=%%a"
if not defined NODE_MAJOR (
    echo ERROR: Could not detect Node.js version
    echo Please install Node.js 18 or higher
    pause
    exit /b 1
)
set /A NODE_MAJOR_NUM=!NODE_MAJOR! 2>nul
if !NODE_MAJOR_NUM! LSS 18 (
    echo ERROR: Node.js version is less than 18; please upgrade to Node.js 18 or higher
    pause
    exit /b 1
)

REM Candidate frontend ports (space-separated). Can be overridden via FRONTEND_CANDIDATE_PORTS env var
if defined FRONTEND_CANDIDATE_PORTS (
    set "CANDIDATE_PORTS=%FRONTEND_CANDIDATE_PORTS%"
) else (
    set "CANDIDATE_PORTS=3000 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010"
)
set "PORT="
for %%p in (%CANDIDATE_PORTS%) do (
    REM Match exact port token and exclude TIME_WAIT entries to avoid false positives like 3000 matching 30001
    netstat -ano | findstr ":%%p " | findstr /V "TIME_WAIT" >nul
    if errorlevel 1 (
        set "PORT=%%p"
        goto :frontend_port_found
    ) else (
        echo Port %%p is in use; trying next...
    )
)

echo ERROR: No available frontend port found (tried: %CANDIDATE_PORTS%)
exit /b 1

:frontend_port_found
echo Selected frontend port: !PORT!

REM Candidate backend ports (space-separated). Can be overridden via BACKEND_CANDIDATE_PORTS env var
if defined BACKEND_CANDIDATE_PORTS (
    set "BACKEND_PORTS=%BACKEND_CANDIDATE_PORTS%"
) else (
    set "BACKEND_PORTS=8000 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010"
)
set "BACKEND_PORT="
for %%b in (%BACKEND_PORTS%) do (
    REM Use regex-style search for ":<port> " and filter out TIME_WAIT entries to avoid false positives
    netstat -ano | findstr /R ":%%b " | findstr /V "TIME_WAIT" >nul
    if errorlevel 1 (
        set "BACKEND_PORT=%%b"
        goto :backend_port_found
    ) else (
        echo Backend port %%b is in use; trying next...
    )
)

echo ERROR: No available backend port found (tried: %BACKEND_PORTS%)
exit /b 1

:backend_port_found
echo Selected backend port: !BACKEND_PORT!

REM Set runtime configuration
set FRONTEND_URL=http://localhost:!PORT!
set NEXT_PUBLIC_API_URL=http://localhost:!BACKEND_PORT!
set WARMUP_PREVIEW=true
REM Default backend host (can be overridden in environment): binds uvicorn to this host if not provided
if not defined BACKEND_HOST set "BACKEND_HOST=0.0.0.0"

echo.
echo Starting VibeVoice-Narrator Web GUI...
echo Frontend will be available at: http://localhost:!PORT!
echo Backend API will be available at: http://localhost:!BACKEND_PORT!
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start backend + frontend in Windows Terminal tabs if available
where wt >nul 2>&1
if !errorlevel!==0 (
    REM Start backend in its own Windows Terminal tab
    wt -w 0 new-tab --title "VibeVoice Backend" -d "%ROOT%\backend" cmd /k "set FRONTEND_URL=http://localhost:!PORT! && set NEXT_PUBLIC_API_URL=http://localhost:!BACKEND_PORT! && set WARMUP_PREVIEW=!WARMUP_PREVIEW! && set BACKEND_HOST=!BACKEND_HOST! && set BACKEND_PORT=!BACKEND_PORT! && python -m uvicorn main:app --host !BACKEND_HOST! --port !BACKEND_PORT! --reload"

    REM Wait for backend health before launching frontend (callable loop)
    if not defined BACKEND_HEALTH_TIMEOUT_SECONDS set BACKEND_HEALTH_TIMEOUT_SECONDS=15
    set /A attempts=0
    call :wait_backend_loop start_frontend_wt

    :start_frontend_wt
    wt -w 0 new-tab --title "VibeVoice Frontend" -d "%ROOT%\frontend" cmd /k "set NEXT_PUBLIC_API_URL=http://localhost:!BACKEND_PORT! && set WARMUP_PREVIEW=!WARMUP_PREVIEW! && npm run dev -- --port !PORT!"
    start "" "http://localhost:!PORT!"
) else (
    REM Fallback to separate windows
    start "VibeVoice Backend" cmd /k "set FRONTEND_URL=http://localhost:!PORT! && set NEXT_PUBLIC_API_URL=http://localhost:!BACKEND_PORT! && set WARMUP_PREVIEW=!WARMUP_PREVIEW! && set BACKEND_HOST=!BACKEND_HOST! && set BACKEND_PORT=!BACKEND_PORT! && cd backend && python -m uvicorn main:app --host !BACKEND_HOST! --port !BACKEND_PORT! --reload"

    REM Wait for backend health before launching frontend window (callable loop)
    if not defined BACKEND_HEALTH_TIMEOUT_SECONDS set BACKEND_HEALTH_TIMEOUT_SECONDS=15
    set /A attempts=0
    call :wait_backend_loop start_frontend_fallback

    :start_frontend_fallback
    start "VibeVoice Frontend" cmd /k "set NEXT_PUBLIC_API_URL=http://localhost:!BACKEND_PORT! && set WARMUP_PREVIEW=!WARMUP_PREVIEW! && cd frontend && npm run dev -- --port !PORT!"

REM Wait-loop subroutine for backend health checks
:wait_backend_loop
REM Usage: call :wait_backend_loop <label-to-goto-on-success>
setlocal enabledelayedexpansion
if "%~1"=="" (
    echo INTERNAL ERROR: wait_backend_loop requires a target label
    endlocal
    exit /b 1
)
:wait_backend_loop_start
if !attempts! GEQ !BACKEND_HEALTH_TIMEOUT_SECONDS! (
    echo ERROR: Backend did not become healthy within !BACKEND_HEALTH_TIMEOUT_SECONDS! seconds
    endlocal
    exit /b 1
)
set /A attempts+=1
where curl >nul 2>&1
if !errorlevel!==0 (
    curl --silent --fail --max-time 2 http://localhost:!BACKEND_PORT!/health >nul 2>&1
    if !errorlevel!==0 (
        echo Backend is healthy
        endlocal
        goto %~1
    )
) else (
    powershell -Command "try{ $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:!BACKEND_PORT!/health' -TimeoutSec 2; exit 0 } catch { exit 1 }"
    if !errorlevel!==0 (
        echo Backend is healthy
        endlocal
        goto %~1
    )
)
timeout /t 1 /nobreak >nul
goto wait_backend_loop_start

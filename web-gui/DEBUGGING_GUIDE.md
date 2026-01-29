# VibeVoice-Narrator Web GUI Debugging Guide

This guide provides a comprehensive debugging strategy for troubleshooting the [`start.bat`](start.bat) script when it fails to launch the web GUI.

---

## Table of Contents

1. [Quick Diagnosis Steps](#quick-diagnosis-steps)
2. [Command Line Execution for Error Capture](#command-line-execution-for-error-capture)
3. [System Prerequisites Checklist](#system-prerequisites-checklist)
4. [File Path and Permission Verification](#file-path-and-permission-verification)
5. [Common Issues and Solutions](#common-issues-and-solutions)
6. [Advanced Debugging](#advanced-debugging)

---

## Quick Diagnosis Steps

Before diving into detailed debugging, perform these quick checks:

1. **Double-click the script** - Note any error messages that appear briefly
2. **Check Task Manager** - Look for Python/Node processes that may be running
3. **Check browser** - Try accessing `http://localhost:3000` or `http://localhost:8000`

---

## Command Line Execution for Error Capture

### Method 1: Run from Command Prompt (Recommended)

This method captures all output and prevents the window from closing immediately.

```cmd
REM Open Command Prompt (cmd.exe) and navigate to the project directory:
cd /d "<path-to-project>"

REM Run the script directly:
start.bat

REM OR run with verbose output by modifying the script temporarily:
REM Add "echo on" as the second line of start.bat
```

### Method 2: Run with Output Redirection

```cmd
cd /d "<path-to-project>"
start.bat > debug_output.log 2>&1
type debug_output.log
```

### Method 3: Step-by-Step Manual Execution

Execute each command from [`start.bat`](start.bat) manually to isolate the failing step:

```cmd
REM 1. Check Python installation
python --version

REM 2. Check Node.js installation
node --version

REM 3. Check npm installation
npm --version

REM 4. Navigate to backend directory
cd backend

REM 5. Try starting backend manually
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

REM 6. In a NEW terminal, navigate to frontend directory
cd /d "<path-to-project>\frontend"

REM 7. Try starting frontend manually
npm run dev -- --port 3000
```

### Method 4: Run with PowerShell for Better Error Handling

```powershell
# Open PowerShell and navigate to project
cd "<path-to-project>"
# Run start.bat and check its exit code
& .\start.bat
if ($LASTEXITCODE -ne 0) {
    Write-Host "start.bat exited with code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "start.bat completed successfully" -ForegroundColor Green
}
```

---

## System Prerequisites Checklist

### Python Environment

- [ ] **Python 3.9 or higher** installed
  - Verify: `python --version` or `python3 --version`
  - Download: https://www.python.org/downloads/
  - **Important**: During installation, check "Add Python to PATH"

- [ ] **Python packages installed** (from [`backend/requirements.txt`](backend/requirements.txt))
  ```cmd
  cd backend
  pip install -r requirements.txt
  # For development and running tests, also install dev requirements:
  pip install -r dev-requirements.txt  # preferred (pinned)
  # or
  pip install -r dev-requirements.in
  ```
  Required packages:
  - fastapi==0.115.0
  - uvicorn[standard]==0.30.6
  - pydantic==2.9.2
  - pydantic-settings==2.6.1
  - python-multipart==0.0.9
  - aiofiles==24.1.0
  - sqlalchemy==2.0.35
  - aiosqlite==0.20.0

- [ ] **VibeVoice model** (optional but recommended for full functionality)
  - The script uses `microsoft/VibeVoice-Realtime-0.5B` by default
  - Will be downloaded automatically on first use if internet is available

### Node.js Environment

- [ ] **Node.js 18 or higher** installed
  - Verify: `node --version`
  - Download: https://nodejs.org/

- [ ] **npm** (comes with Node.js)
  - Verify: `npm --version`

- [ ] **Node.js dependencies installed** (from [`frontend/package.json`](frontend/package.json))
  ```cmd
  cd frontend
  npm install
  ```
  Key dependencies:
  - next: 16.1.4
  - react: 19.2.3
  - react-dom: 19.2.3
  - Various UI components (@radix-ui/*)

### System Requirements

- [ ] **Available ports**: 3000 (frontend) and 8000 (backend) must be free
  - Check: `netstat -ano | findstr ":3000"` and `netstat -ano | findstr ":8000"`
  - If occupied, the script will try port 3001 for frontend

- [ ] **Disk space**: At least 500MB free for dependencies and model files

- [ ] **Network access**: Required for:
  - Downloading Python packages (PyPI)
  - Downloading Node.js packages (npm registry)
  - Downloading VibeVoice model (Hugging Face)

- [ ] **Windows permissions**: Ability to:
  - Create directories in `web-gui/data/`
  - Write to `web-gui/data/audio/`
  - Execute Python and Node.js scripts

---

## File Path and Permission Verification

### Directory Structure Verification

Verify the following directory structure exists:

```
VibeVoice-Narrator/
└── web-gui/
    ├── start.bat
    ├── backend/
    │   ├── main.py
    │   ├── config.py
    │   ├── requirements.txt
    │   └── routes/
    ├── frontend/
    │   ├── package.json
    │   ├── next.config.ts
    │   ├── src/
    │   └── node_modules/  (created after npm install)
    └── data/
        ├── audio/         (auto-created by config.py)
        ├── documents/     (auto-created by config.py)
        └── voices/        (auto-created by config.py)
```

### Verification Commands

```cmd
REM Check if all required files exist
cd /d "<path-to-project>"

REM Backend files
if exist "backend\main.py" (echo [OK] backend\main.py exists) else (echo [MISSING] backend\main.py)
if exist "backend\config.py" (echo [OK] backend\config.py exists) else (echo [MISSING] backend\config.py)
if exist "backend\requirements.txt" (echo [OK] backend\requirements.txt exists) else (echo [MISSING] backend\requirements.txt)

REM Frontend files
if exist "frontend\package.json" (echo [OK] frontend\package.json exists) else (echo [MISSING] frontend\package.json)
if exist "frontend\next.config.ts" (echo [OK] frontend\next.config.ts exists) else (echo [MISSING] frontend\next.config.ts)

REM Data directories (should be auto-created)
if exist "data\audio" (echo [OK] data\audio exists) else (echo [MISSING] data\audio - will be created)
if exist "data\documents" (echo [OK] data\documents exists) else (echo [MISSING] data\documents - will be created)
if exist "data\voices" (echo [OK] data\voices exists) else (echo [MISSING] data\voices - will be created)
```

### Permission Verification

```cmd
REM Test write permissions in data directory
cd /d "<path-to-project>"
echo test > data\permission_test.txt
if exist "data\permission_test.txt" (
    echo [OK] Write permissions are valid
    del data\permission_test.txt
) else (
    echo [ERROR] No write permission in data directory
)

REM Test Python execution
python -c "print('Python execution OK')"

REM Test Node.js execution
node -e "console.log('Node.js execution OK')"
```

### Path Issues

The script uses relative paths that depend on the current working directory:

1. **Backend path**: `cd backend` - assumes script is run from `web-gui/` directory
2. **Frontend path**: `cd frontend` - assumes script is run from `web-gui/` directory
3. **Data paths**: Configured in [`backend/config.py`](backend/config.py) using `Path(__file__).parent.parent`

**To fix path issues:**

```cmd
REM Always run the script from the web-gui directory
cd /d "<path-to-project>"
start.bat
```

---

## Common Issues and Solutions

### Issue 1: "Python is not installed or not in PATH"

**Symptoms**: Script exits with error at line 10

**Diagnosis**:
```cmd
python --version
```

**Solutions**:
1. Install Python 3.9+ from https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. If already installed, add to PATH manually:
   - Find Python installation: `where python`
   - Add to System PATH: `C:\Python39\` and `C:\Python39\Scripts\`
4. Restart Command Prompt after PATH changes

### Issue 2: "Node.js is not installed or not in PATH"

**Symptoms**: Script exits with error at line 19

**Diagnosis**:
```cmd
node --version
```

**Solutions**:
1. Install Node.js 18+ from https://nodejs.org/
2. Restart Command Prompt after installation
3. Verify npm is also installed: `npm --version`

### Issue 3: "ModuleNotFoundError" for Python packages

**Symptoms**: Backend fails to start with import errors

**Diagnosis**:
```cmd
cd backend
python -c "import fastapi; import uvicorn; import pydantic"
```

**Solutions**:
```cmd
cd backend
pip install -r requirements.txt
```

### Issue 4: "npm ERR!" or missing node_modules

**Symptoms**: Frontend fails to start with module not found errors

**Diagnosis**:
```cmd
cd frontend
dir node_modules
```

**Solutions**:
```cmd
cd frontend
npm install
```

### Issue 5: Port Already in Use

**Symptoms**: Backend or frontend fails to start with "Address already in use"

**Diagnosis**:
```cmd
netstat -ano | findstr ":3000"
netstat -ano | findstr ":8000"
```

**Solutions**:
1. Kill the process using the port:
   ```cmd
   REM Find PID from netstat output, then:
   taskkill /PID <PID> /F
   ```
2. Or use alternative ports by modifying the script

### Issue 6: CORS Errors in Browser

**Symptoms**: Frontend can't connect to backend API

**Diagnosis**: Check browser console for CORS errors

**Solutions**:
1. Verify [`backend/config.py`](backend/config.py) has correct `FRONTEND_URL`
2. Ensure both servers are running
3. Check firewall settings

### Issue 7: Missing Data Directories

**Symptoms**: Backend fails to start with directory not found errors

**Diagnosis**:
```cmd
dir web-gui\data
```

**Solutions**:
The directories should be auto-created by [`backend/config.py`](backend/config.py). If not:
```cmd
mkdir web-gui\data\audio
mkdir web-gui\data\documents
mkdir web-gui\data\voices
```

---

## Advanced Debugging

### Enable Verbose Logging

Modify [`start.bat`](start.bat) to add debug output. Replace the existing `@echo off` line with `@echo on` and add `set DEBUG=1` if not already present. Use REM comments to make the edit explicit, for example:

```batch
REM File: start.bat
REM Change this line:
REM   @echo off
REM To this:
REM   @echo on
REM Then add this line (if not already present):
REM   set DEBUG=1
```

### Check Backend Logs

The backend runs in a separate window. Check that window for:
- Import errors
- Database connection issues
- Model loading errors

### Check Frontend Logs

The frontend runs in the main terminal. Look for:
- Build errors
- TypeScript errors
- Missing dependencies

### Browser Developer Tools

1. Open browser to `http://localhost:3000`
2. Press F12 to open Developer Tools
3. Check Console tab for JavaScript errors
4. Check Network tab for failed API requests

### Test Backend API Directly

```cmd
REM Test health endpoint
curl http://localhost:8000/health

REM Test voices endpoint
curl http://localhost:8000/voices

REM Test config endpoint
curl http://localhost:8000/config
```

### Check Python Virtual Environment

If using a virtual environment:

```cmd
REM Activate virtual environment
cd backend
venv\Scripts\activate

REM Verify packages
pip list

REM Run backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Diagnostic Script

Save this as `diagnose.bat` in the `web-gui` directory and run it:

```batch
@echo off
echo ========================================
echo   VibeVoice-Narrator Diagnostic Tool
echo ========================================
echo.

echo [1] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python not found in PATH
) else (
    echo [OK] Python found
    python --version
)
echo.

echo [2] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Node.js not found in PATH
) else (
    echo [OK] Node.js found
    node --version
)
echo.

echo [3] Checking npm...
npm --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] npm not found in PATH
) else (
    echo [OK] npm found
    npm --version
)
echo.

echo [4] Checking Python packages...
cd backend
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] fastapi not installed
) else (
    echo [OK] fastapi installed
)
python -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] uvicorn not installed
) else (
    echo [OK] uvicorn installed
)
cd ..
echo.

echo [5] Checking Node.js dependencies...
if exist "frontend\node_modules" (
    echo [OK] node_modules exists
) else (
    echo [FAIL] node_modules not found - run 'npm install' in frontend directory
)
echo.

echo [6] Checking ports...
netstat -ano | findstr ":3000" >nul
if errorlevel 1 (
    echo [OK] Port 3000 is available
) else (
    echo [WARN] Port 3000 is in use
)
netstat -ano | findstr ":8000" >nul
if errorlevel 1 (
    echo [OK] Port 8000 is available
) else (
    echo [WARN] Port 8000 is in use
)
echo.

echo [7] Checking data directories...
if exist "data\audio" (
    echo [OK] data\audio exists
) else (
    echo [WARN] data\audio missing - will be auto-created
)
if exist "data\documents" (
    echo [OK] data\documents exists
) else (
    echo [WARN] data\documents missing - will be auto-created
)
if exist "data\voices" (
    echo [OK] data\voices exists
) else (
    echo [WARN] data\voices missing - will be auto-created
)
echo.

echo ========================================
echo   Diagnostic Complete
echo ========================================
pause
```

---

## Next Steps

After running through this guide:

1. **If all checks pass**: Run [`start.bat`](start.bat) again
2. **If issues found**: Follow the specific solution for each issue
3. **If still failing**: Collect the following information:
   - Output from `diagnose.bat`
   - Error messages from [`start.bat`](start.bat)
   - Browser console errors (if frontend loads)
   - Backend window output (if backend starts)

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [VibeVoice Project README](../README.md)
- [Web GUI Proposal](../docs/web-gui-proposal.md)

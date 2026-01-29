@echo off
echo ========================================
echo   VibeVoice-Narrator Diagnostic Tool
echo ========================================
echo.

echo [1] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python not found in PATH
    echo       Please install Python 3.9 or higher from https://www.python.org/downloads/
    echo       Make sure to check "Add Python to PATH" during installation
) else (
    echo [OK] Python found
    python --version
)
echo.

echo [2] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Node.js not found in PATH
    echo       Please install Node.js 18 or higher from https://nodejs.org/
) else (
    echo [OK] Node.js found
    node --version
)
echo.

echo [3] Checking npm...
npm --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] npm not found in PATH
    echo       npm should be installed with Node.js
) else (
    echo [OK] npm found
    npm --version
)
echo.

echo [4] Checking Python packages...
cd backend >nul 2>&1
if errorlevel 1 (
    echo [FAIL] backend directory not found
    echo       Please ensure you are running this script from the web-gui directory
) else (
    python -c "import fastapi" >nul 2>&1
    if errorlevel 1 (
        echo [FAIL] fastapi not installed
        echo       Run: cd backend ^&^& pip install -r requirements.txt && pip install -r dev-requirements.txt  # for tests/dev
    ) else (
        echo [OK] fastapi installed
    )
    python -c "import uvicorn" >nul 2>&1
    if errorlevel 1 (
        echo [FAIL] uvicorn not installed
        echo       Run: cd backend ^&^& pip install -r requirements.txt && pip install -r dev-requirements.txt  # for tests/dev
    ) else (
        echo [OK] uvicorn installed
    )
    python -c "import pydantic" >nul 2>&1
    if errorlevel 1 (
        echo [FAIL] pydantic not installed
        echo       Run: cd backend ^&^& pip install -r requirements.txt && pip install -r dev-requirements.txt  # for tests/dev
    ) else (
        echo [OK] pydantic installed
    )
    python -c "import pydantic_settings" >nul 2>&1
    if errorlevel 1 (
        echo [FAIL] pydantic-settings not installed
        echo       Run: cd backend ^&^& pip install -r requirements.txt && pip install -r dev-requirements.txt  # for tests/dev
    ) else (
        echo [OK] pydantic-settings installed
    )
    cd .. >nul 2>&1
)
echo.

echo [5] Checking Node.js dependencies...
if exist "frontend\node_modules" (
    echo [OK] node_modules exists
) else (
    echo [FAIL] node_modules not found
    echo       Run: cd frontend ^&^& npm install
)
echo.

echo [6] Checking ports...
netstat -ano | findstr /R ":3000[^0-9]" >nul 2>&1
if errorlevel 1 (
    echo [OK] Port 3000 is available
) else (
    echo [WARN] Port 3000 is in use
    echo       The script will try to use port 3001 instead
)
netstat -ano | findstr /R ":8000[^0-9]" >nul 2>&1
if errorlevel 1 (
    echo [OK] Port 8000 is available
) else (
    echo [WARN] Port 8000 is in use
    echo       Backend may fail to start. Kill the process using this port:
    echo       netstat -ano ^| findstr ":8000[^0-9]"
    echo       taskkill /PID ^<PID^> /F
)
echo.

echo [7] Checking data directories...
if exist "data\audio" (
    echo [OK] data\audio exists
) else (
    echo [WARN] data\audio missing - will be auto-created by backend
)
if exist "data\documents" (
    echo [OK] data\documents exists
) else (
    echo [WARN] data\documents missing - will be auto-created by backend
)
if exist "data\voices" (
    echo [OK] data\voices exists
) else (
    echo [WARN] data\voices missing - will be auto-created by backend
)
echo.

echo [8] Checking required files...
if exist "backend\main.py" (
    echo [OK] backend\main.py exists
) else (
    echo [FAIL] backend\main.py missing
)
if exist "backend\config.py" (
    echo [OK] backend\config.py exists
) else (
    echo [FAIL] backend\config.py missing
)
if exist "backend\requirements.txt" (
    echo [OK] backend\requirements.txt exists
) else (
    echo [FAIL] backend\requirements.txt missing
)
if exist "frontend\package.json" (
    echo [OK] frontend\package.json exists
) else (
    echo [FAIL] frontend\package.json missing
)
if exist "frontend\next.config.ts" (
    echo [OK] frontend\next.config.ts exists
) else (
    echo [FAIL] frontend\next.config.ts missing
)
echo.

echo [9] Testing write permissions...
if not exist "data" (
    echo [INFO] data directory missing, creating it...
    mkdir "data" 2>nul || echo [WARN] Could not create data directory
)
echo test > data\permission_test.txt 2>nul
if exist "data\permission_test.txt" (
    echo [OK] Write permissions are valid
    del data\permission_test.txt >nul 2>&1
) else (
    echo [FAIL] No write permission in data directory or data directory missing
    echo       Check folder permissions and ensure you have rights to create/write to 'data' directory
)
echo.

echo ========================================
echo   Diagnostic Complete
echo ========================================
echo.
echo If all checks show [OK], you should be able to run start.bat successfully.
echo If you see [FAIL] or [WARN], please follow the suggested fixes above.
echo.
pause

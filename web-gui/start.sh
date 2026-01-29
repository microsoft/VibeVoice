#!/bin/bash

echo "========================================"
echo "  VibeVoice-Narrator Web GUI"
echo "========================================"
echo ""
# Resolve script dir so we can use absolute paths when changing directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python is not installed"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed"
    echo "Please install Node.js 18 or higher"
    exit 1
fi

# Port probing helper: try lsof, then ss, then netstat
is_port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    # Quote the port variable to avoid word-splitting or globbing
    lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1 && return 0 || return 1
  elif command -v ss >/dev/null 2>&1; then
    # ss prints addresses like "0.0.0.0:3000" or "[::]:3000" in the 4th column
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -E ":[[:digit:]]+$" | grep -E ":${port}$" >/dev/null 2>&1 && return 0 || return 1
  elif command -v netstat >/dev/null 2>&1; then
    netstat -tln 2>/dev/null | awk '{print $4}' | grep -E ":[[:digit:]]+$" | grep -E ":${port}$" >/dev/null 2>&1 && return 0 || return 1
  else
    # If we cannot detect, assume not in use (best-effort)
    echo "WARNING: Could not detect ports (no lsof/ss/netstat). Assuming port ${port} may be free."
    return 1
  fi
}

# Candidate ports: allow override via FRONTEND_CANDIDATE_PORTS env (comma or space separated)
if [ -n "$FRONTEND_CANDIDATE_PORTS" ]; then
  # Normalize commas to spaces and split on whitespace so both comma- and space-separated lists work
  NORMALIZED=$(echo "$FRONTEND_CANDIDATE_PORTS" | tr ',' ' ')
  read -r -a TMP_ARRAY <<< "$NORMALIZED"
  # Validate tokens: accept only numeric port tokens within 1-65535
  CANDIDATE_PORTS=()
  for token in "${TMP_ARRAY[@]}"; do
    if [ -z "$token" ]; then
      continue
    fi
    # Numeric check
    if [[ "$token" =~ ^[0-9]+$ ]]; then
      # Range check (valid TCP/UDP port range)
      if [ "$token" -ge 1 ] && [ "$token" -le 65535 ]; then
        CANDIDATE_PORTS+=("$token")
      else
        echo "WARNING: Skipping invalid port (out of range): $token"
      fi
    else
      echo "WARNING: Skipping non-numeric port token: $token"
    fi
  done

  # If all tokens were invalid, fall back to default range
  if [ ${#CANDIDATE_PORTS[@]} -eq 0 ]; then
    echo "WARNING: No valid FRONTEND_CANDIDATE_PORTS provided; falling back to default ports 3000-3010"
    CANDIDATE_PORTS=( {3000..3010} )
  fi
else
  # Default range 3000-3010
  CANDIDATE_PORTS=( {3000..3010} )
fi

PORT=""
for p in "${CANDIDATE_PORTS[@]}"; do
  if ! is_port_in_use "$p"; then
    PORT=$p
    break
  fi
done
if [ -z "$PORT" ]; then
  echo "ERROR: No available frontend port found (tried: ${CANDIDATE_PORTS[*]})"
  exit 1
fi

# Set runtime configuration
export FRONTEND_URL="http://localhost:$PORT"
export WARMUP_PREVIEW="true"

# Ensure backend port is available
BACKEND_PORT=${BACKEND_PORT:-8000}
if is_port_in_use "$BACKEND_PORT"; then
  echo "ERROR: Backend port $BACKEND_PORT is already in use"
  exit 1
fi

# Now set NEXT_PUBLIC_API_URL using the resolved BACKEND_PORT
export NEXT_PUBLIC_API_URL="http://localhost:$BACKEND_PORT"

echo ""
echo "Starting VibeVoice-Narrator Web GUI..."
echo "Frontend will be available at: http://localhost:$PORT"
echo "Backend API will be available at: http://localhost:$BACKEND_PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Start backend in background (ensure directory exists)
if ! cd "$SCRIPT_DIR/backend"; then
  echo "ERROR: Could not change directory to $SCRIPT_DIR/backend"
  exit 1
fi
# Allow BACKEND_HOST to be configured; default to localhost (127.0.0.1) for safe local development
BACKEND_HOST=${BACKEND_HOST:-127.0.0.1}
python3 -m uvicorn main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!
# Register cleanup trap immediately after backend launch; only kill if process still exists
trap 'if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then kill "$BACKEND_PID"; fi' EXIT

# Wait for backend health endpoint to become available (configurable timeout)
BACKEND_HEALTH_TIMEOUT_SECONDS=${BACKEND_HEALTH_TIMEOUT_SECONDS:-15}
HEALTH_URL="http://localhost:$BACKEND_PORT/health"

echo "Waiting up to ${BACKEND_HEALTH_TIMEOUT_SECONDS}s for backend to become healthy at $HEALTH_URL..."
deadline=$(( $(date +%s) + BACKEND_HEALTH_TIMEOUT_SECONDS ))
HEALTHY=""
while (( $(date +%s) <= deadline )); do
  if command -v curl >/dev/null 2>&1; then
    if curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
      echo "Backend is healthy"
      HEALTHY=1
      break
    fi
  elif command -v wget >/dev/null 2>&1; then
    if wget --quiet --spider --timeout=2 "$HEALTH_URL" >/dev/null 2>&1; then
      echo "Backend is healthy"
      HEALTHY=1
      break
    fi
  else
    # Fallback to Python HTTP check with short timeout
    if python3 - <<PY >/dev/null 2>&1
import sys, http.client, urllib.parse
u = urllib.parse.urlparse("$HEALTH_URL")
conn = http.client.HTTPConnection(u.hostname, u.port, timeout=2)
try:
    conn.request("GET", u.path or "/")
    r = conn.getresponse()
    # Treat only 2xx and 3xx responses as healthy; 4xx client errors should be treated as failures
    sys.exit(0 if 200 <= r.status < 400 else 1)
except Exception:
    sys.exit(1)
finally:
    conn.close()
PY
    then
      echo "Backend is healthy"
      HEALTHY=1
      break
    fi
  fi
  sleep 0.5
done

if [ -z "$HEALTHY" ]; then
  echo "ERROR: Backend did not become healthy within ${BACKEND_HEALTH_TIMEOUT_SECONDS}s at $HEALTH_URL"
  # Attempt to kill backend process if still running
  if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    echo "Killing backend process $BACKEND_PID"
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  exit 1
fi

# Start frontend (ensure directory exists)
if ! cd "$SCRIPT_DIR/frontend"; then
  echo "ERROR: Could not change directory to $SCRIPT_DIR/frontend"
  exit 1
fi
npm run dev -- --port $PORT

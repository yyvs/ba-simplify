#!/usr/bin/env bash

# Runs `uvicorn main:app --reload` and stops it (server + reloader) after
# IDLE_TIMEOUT seconds with no requests or reload events. Activity is read from
# uvicorn's log output. /health requests don't count (the extension polls it on a
# timer, so the timeout would never fire); override with IDLE_IGNORE.
#
# uvicorn runs under `script` (a pty wrapper) so it still sees a terminal: colored
# output, and progress bars (e.g. weight loading) overwrite one line in place.
#
# Also serves `../demo` (the extension's manual test pages) over HTTP, so they work
# without "Allow access to file URLs". Not watched by the idle timer, but stops with
# uvicorn.
#
# Also starts/stops the Ollama sidecar for the "llm_*" models. It must be up
# *before* uvicorn: main.py probes it once at startup. An Ollama already serving
# (brew service, desktop app, another terminal) is reused and left running; only a
# sidecar started here is stopped.
#
# Usage:
#   ./run_dev.sh                  # stop after 10 minutes idle (default)
#   ./run_dev.sh -idle 900        # stop after 15 minutes idle
#   ./run_dev.sh -noidle          # never stop on its own
#   IDLE_TIMEOUT=900 ./run_dev.sh # same as -idle 900, as an env var
#   PORT=8010 ./run_dev.sh        # backend port (default 8000)
#   DEMO_PORT=8011 ./run_dev.sh   # demo-pages port (default 8001)
#   NO_DEMO=1 ./run_dev.sh        # skip serving demo/ entirely
#   NO_LLM=1 ./run_dev.sh         # skip the Ollama sidecar (llm_* unavailable)
#   OLLAMA_URL=... ./run_dev.sh   # sidecar address (must match the backend's)
#   IDLE_IGNORE=... ./run_dev.sh  # log substring that doesn't count as activity
set -uo pipefail

# 0 = never stop on its own (what -noidle sets).
IDLE_TIMEOUT="${IDLE_TIMEOUT:-600}"
PORT="${PORT:-8000}"
DEMO_PORT="${DEMO_PORT:-8001}"
NO_DEMO="${NO_DEMO:-}"
NO_LLM="${NO_LLM:-}"
# Same default as main.py's OLLAMA_URL; exported so both ends use one address.
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
export OLLAMA_URL
# Access-log lines containing this substring don't count as activity (literal match,
# not regex).
IDLE_IGNORE="${IDLE_IGNORE:-GET /health}"
CHECK_INTERVAL=10
# Seconds to wait for a started sidecar to answer /api/tags. A cold `ollama serve`
# needs well under a second; the slack is for a loaded machine.
OLLAMA_WAIT=20

cd "$(dirname "$0")" || exit
DEMO_DIR="../demo"

# Colored output for this script's own messages only; uvicorn/HF output passes
# through `script` untouched. Aligns with uvicorn's "INFO:     " prefix using its
# formula (uvicorn/logging.py DefaultFormatter): category padded to 8 chars, plus
# the one space from "%(levelprefix)s %(message)s", so messages start at column 10.
if [[ -t 1 ]]; then
	BOLD=$'\033[1m'
	YELLOW=$'\033[33m'
	RED=$'\033[31m'
	RESET=$'\033[0m'
else
	BOLD=""
	YELLOW=""
	RED=""
	RESET=""
fi

log() {
	local category="$1" color="$2" message="$3"
	local sep=$((8 - ${#category}))
	[[ ${sep} -lt 0 ]] && sep=0
	message=$(printf '%s' "${message}" | sed -E "s#(https?://[^[:space:]]+)#${BOLD}\\1${RESET}#g")
	printf '%s%s:%*s%s%s\n' "${color}" "${category}" $((sep + 1)) "" "${RESET}" "${message}"
}
log_system() { log "SYSTEM" "${YELLOW}" "$1"; }
log_warning() { log "WARNING" "${RED}" "$1"; }

# Flags override the env vars above; only the idle timeout has one. Parsed here
# because the error paths need log_warning.
usage() {
	cat <<EOF
Usage: ./run_dev.sh [-idle SECONDS | -noidle]

  -idle, --idle SECONDS   stop after SECONDS with no activity (default ${IDLE_TIMEOUT})
  --idle=SECONDS          same, as a single argument
  -noidle, --no-idle      never stop on its own (same as -idle 0)
  -h, --help              show this and exit

Env vars: IDLE_TIMEOUT, IDLE_IGNORE, PORT, DEMO_PORT, NO_DEMO, NO_LLM, OLLAMA_URL
See the comment at the top of this script for what each one does.
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	-idle | --idle)
		if [[ $# -lt 2 ]]; then
			log_warning "$1 needs a value, e.g. \`-idle 900\`." >&2
			exit 1
		fi
		IDLE_TIMEOUT="$2"
		shift 2
		;;
	--idle=*)
		IDLE_TIMEOUT="${1#*=}"
		shift
		;;
	-noidle | --noidle | --no-idle)
		IDLE_TIMEOUT=0
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		log_warning "Unknown option '$1'." >&2
		usage >&2
		exit 1
		;;
	esac
done

case "${IDLE_TIMEOUT}" in
'' | *[!0-9]*)
	log_warning "Idle timeout must be a whole number of seconds, got '${IDLE_TIMEOUT}'." >&2
	exit 1
	;;
*)
	# a run of digits: valid
	;;
esac

# Bytes of activity in the log, excluding IDLE_IGNORE lines. Not mtime (can't tell
# /health from real requests), not a line count (the progress bar rewrites one line
# via carriage returns). Only changes matter, not the absolute value.
active_bytes() {
	awk -v ignore="${IDLE_IGNORE}" '
    index($0, ignore) { next }
    { total += length($0) + 1 }
    END { print total + 0 }
  ' "$1" 2>/dev/null
}

LOGFILE=$(mktemp)
# Declared up front so the cleanup trap exists before anything is spawned (a
# Ctrl-C during the sidecar's startup wait would otherwise leave it running).
SERVER_PID=""
UVICORN_PID=""
DEMO_PID=""
WATCH_PID=""
OLLAMA_PID=""
OLLAMA_LOG=""

stop_ollama() {
	# empty unless started here, so a pre-existing Ollama keeps running
	[[ -n ${OLLAMA_PID} ]] || return 0
	# Each loaded model is a llama-server child holding several GB (keep_alive=-1).
	# Collect children before signalling the parent: once orphaned they can't be
	# told apart from unrelated ones.
	local runners runner
	runners=$(pgrep -P "${OLLAMA_PID}" 2>/dev/null)
	kill -TERM "${OLLAMA_PID}" 2>/dev/null
	wait "${OLLAMA_PID}" 2>/dev/null
	for runner in ${runners}; do
		kill -TERM "${runner}" 2>/dev/null
	done
	OLLAMA_PID=""
	[[ -n ${OLLAMA_LOG} ]] && rm -f "${OLLAMA_LOG}"
	OLLAMA_LOG=""
}

# Cleanup on any exit path. The traps need an explicit `exit`: otherwise after
# INT/TERM bash resumes the loop below with a deleted logfile. `wait` lets uvicorn's
# shutdown output finish before we exit.
stop_server() {
	if [[ -n ${SERVER_PID} ]]; then
		if [[ -n ${UVICORN_PID} ]]; then
			kill -TERM "${UVICORN_PID}" 2>/dev/null
		else
			kill "${SERVER_PID}" 2>/dev/null
		fi
		wait "${SERVER_PID}" 2>/dev/null
	fi
	[[ -n ${DEMO_PID} ]] && kill "${DEMO_PID}" 2>/dev/null
	[[ -n ${WATCH_PID} ]] && kill "${WATCH_PID}" 2>/dev/null
	# after uvicorn, so its shutdown can still reach the sidecar if it needs to
	stop_ollama
	rm -f "${LOGFILE}"
}
trap 'stop_server; exit 0' EXIT TERM
# The terminal echoes "^C" without a newline; print one so uvicorn's first
# shutdown line doesn't land on the same line.
trap 'echo; stop_server; exit 0' INT

ollama_up() {
	curl -fsS --max-time 2 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1
}

# Before uvicorn: main.py probes the sidecar once at startup, and a model failing
# that probe stays unavailable for the whole run.
start_ollama() {
	if [[ -n ${NO_LLM} ]]; then
		log_system "NO_LLM set — skipping the Ollama sidecar (llm_* models unavailable)."
		return
	fi
	if ollama_up; then
		log_system "Reusing the Ollama already serving on ${OLLAMA_URL} (left running on exit)."
		return
	fi
	if ! command -v ollama >/dev/null 2>&1; then
		log_warning "ollama not found on PATH — llm_* models will be unavailable. Install it, or set NO_LLM=1 to skip this check." >&2
		return
	fi
	OLLAMA_LOG=$(mktemp)
	# Not in $LOGFILE: the idle timer reads that, and sidecar output would count
	# as activity.
	#
	# OLLAMA_HOST is host:port without scheme, derived from the backend's URL.
	OLLAMA_HOST="${OLLAMA_URL#*://}" ollama serve >"${OLLAMA_LOG}" 2>&1 &
	OLLAMA_PID=$!
	for _ in $(seq 1 $((OLLAMA_WAIT * 2))); do
		if ollama_up; then
			log_system "Ollama sidecar on ${OLLAMA_URL} (started here, stops with this script)."
			return
		fi
		# died (port taken by something else, bad OLLAMA_HOST, ...): stop waiting
		kill -0 "${OLLAMA_PID}" 2>/dev/null || break
		sleep 0.5
	done
	log_warning "Ollama didn't answer on ${OLLAMA_URL} within ${OLLAMA_WAIT}s — llm_* models will be unavailable. Last lines of its log:" >&2
	tail -n 5 "${OLLAMA_LOG}" | sed -e 's/^/         /' >&2
	stop_ollama
}
start_ollama

# Without watchfiles (tests/requirements-dev.txt) uvicorn falls back to StatReload,
# which ignores excludes and stats every .py under backend/ (17,680 of 17,693 are in
# venv/) several times a second: ~96% of a core. uvicorn's own warning is easy to miss
# among the startup lines, so check here.
if ! venv/bin/python3 -c "import watchfiles" >/dev/null 2>&1; then
	log_warning "watchfiles isn't installed, so --reload will walk all of backend/ (venv included) several times a second and burn a whole CPU core. Fix: venv/bin/pip install -r tests/requirements-dev.txt" >&2
fi

# -q: suppress script's "Script started/done" banners
# -t 1: flush every 1s (default 30s is too slow for a 10s CHECK_INTERVAL)
# --reload-exclude must be an absolute directory. uvicorn's FileFilter
# (uvicorn/supervisors/watchfilesreload.py) treats an existing directory via
# `exclude_dir in path.parents` and anything else as a Path.match glob, which anchors
# at the path's end: 'venv/*' matches only direct children. A relative `venv` fails
# because watchfiles yields absolute paths.
script -q -t 1 "${LOGFILE}" \
	venv/bin/python3 -m uvicorn main:app --reload --port "${PORT}" \
	--reload-exclude "${PWD}/venv" &
SERVER_PID=$!

# `script` puts its child (the uvicorn reloader) in a new session on its own pty.
# Killing `script` only closes the pty (SIGHUP), skipping uvicorn's graceful
# shutdown, so find and signal the child directly.
for _ in $(seq 1 50); do
	UVICORN_PID=$(pgrep -P "${SERVER_PID}" | head -n1)
	[[ -n ${UVICORN_PID} ]] && break
	sleep 0.1
done

if [[ -z ${NO_DEMO} ]] && [[ -d ${DEMO_DIR} ]]; then
	# silenced: uvicorn's access log already shows the demo pages' /simplify calls
	venv/bin/python3 -m http.server "${DEMO_PORT}" --directory "${DEMO_DIR}" \
		>/dev/null 2>&1 &
	DEMO_PID=$!
elif [[ -z ${NO_DEMO} ]]; then
	log_warning "${DEMO_DIR} not found — skipping the demo server." >&2
fi

# Prints one "ready" message after uvicorn's "Application startup complete";
# once per run, even across --reload restarts.
watch_ready() {
	while kill -0 "${SERVER_PID}" 2>/dev/null; do
		if grep -q "Application startup complete" "${LOGFILE}" 2>/dev/null; then
			if [[ -n ${DEMO_PID} ]]; then
				log_system "Everything's ready — head to http://127.0.0.1:${DEMO_PORT}/ to test the extension."
			else
				log_system "Everything's ready — backend is live on http://127.0.0.1:${PORT}."
			fi
			return
		fi
		sleep 0.5
	done
}
watch_ready &
WATCH_PID=$!

if [[ ${IDLE_TIMEOUT} -eq 0 ]]; then
	log_system "Backend on http://127.0.0.1:${PORT} (no idle timeout — runs until you stop it with Ctrl-C)"
else
	log_system "Backend on http://127.0.0.1:${PORT} (auto-stops after ${IDLE_TIMEOUT}s idle, not counting ${IDLE_IGNORE} — override with -idle ..., PORT=...)"
fi
[[ -n ${DEMO_PID} ]] && log_system "Demo pages on http://127.0.0.1:${DEMO_PORT}/"

if [[ ${IDLE_TIMEOUT} -eq 0 ]]; then
	# no polling; the INT/TERM traps still fire when `wait` is interrupted
	wait "${SERVER_PID}" 2>/dev/null
else
	LAST_BYTES=$(active_bytes "${LOGFILE}")
	LAST_ACTIVITY=$(date +%s)

	while kill -0 "${SERVER_PID}" 2>/dev/null; do
		sleep "${CHECK_INTERVAL}"
		NOW=$(date +%s)
		BYTES=$(active_bytes "${LOGFILE}")
		if [[ ${BYTES} != "${LAST_BYTES}" ]]; then
			LAST_BYTES="${BYTES}"
			LAST_ACTIVITY="${NOW}"
		fi
		IDLE=$((NOW - LAST_ACTIVITY))
		if [[ ${IDLE} -ge ${IDLE_TIMEOUT} ]]; then
			log_system "No activity for ${IDLE}s (>= ${IDLE_TIMEOUT}s) — stopping."
			break
		fi
	done
fi

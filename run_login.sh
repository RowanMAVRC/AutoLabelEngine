#!/usr/bin/env bash

DEFAULT_VENV_PATH="../envs/auto-label-engine"
VENV_PATH="$DEFAULT_VENV_PATH"

# Kill existing ngrok processes if ngrok is installed
if command -v ngrok >/dev/null; then
    if pgrep -x ngrok >/dev/null; then
        echo "Killing existing ngrok processes..."
        pkill -f ngrok
    fi
fi

# Ensure virtual environment exists
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Virtual environment not found."
    read -p "Would you like to generate it? (y/n): " generate_choice
    if [[ ! "$generate_choice" =~ ^[Yy]$ ]]; then
        echo "Exiting without generating virtual environment."
        exit 1
    fi
    while true; do
        read -p "Enter the directory for the virtual environment [default: $DEFAULT_VENV_PATH]: " input_path
        if [ -z "$input_path" ]; then
            input_path="$DEFAULT_VENV_PATH"
        fi
        read -p "You entered '$input_path'. Is this correct? (y/n): " confirm_choice
        if [[ "$confirm_choice" =~ ^[Yy]$ ]]; then
            VENV_PATH="$input_path"
            break
        else
            echo "Let's try again."
        fi
    done
    echo "Setting up virtual environment at '$VENV_PATH'..."
    bash scripts/setup_venv.sh "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

STREAMLIT_PORT=8501

echo "Starting login server on port $STREAMLIT_PORT..."
streamlit run --server.headless True --server.fileWatcherType none \
    --server.port $STREAMLIT_PORT login.py &
STREAMLIT_PID=$!

if command -v ngrok >/dev/null; then
    echo "Starting ngrok tunnel..."
    ngrok http $STREAMLIT_PORT --log=stdout > ngrok_login.log &
    NGROK_PID=$!
    sleep 2
    echo "Your public ngrok URL is:"
    curl --silent http://127.0.0.1:4040/api/tunnels \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
else
    echo "ngrok not installed—running Streamlit locally only."
fi

wait $STREAMLIT_PID
[ -n "$NGROK_PID" ] && wait $NGROK_PID

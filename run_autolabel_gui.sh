#!/usr/bin/env bash

# Default virtual environment directory
DEFAULT_VENV_PATH="../envs/auto-label-engine"
VENV_PATH="$DEFAULT_VENV_PATH"


# === Check for virtual environment ===
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Virtual environment not found."

    read -p "Would you like to generate it? (y/n): " generate_choice
    if [[ ! "$generate_choice" =~ ^[Yy]$ ]]; then
        echo "Exiting without generating virtual environment."
        exit 1
    fi

    # Ask user for the virtual environment directory, using the default if none is provided
    while true; do
        read -p "Enter the directory for the virtual environment [default: $DEFAULT_VENV_PATH]: " input_path
        if [ -z "$input_path" ]; then
            input_path="$DEFAULT_VENV_PATH"
        fi

        # Confirm the entered directory
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

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Ensure Streamlit uses port 8501 unless overridden
STREAMLIT_PORT=${STREAMLIT_PORT:-8501}

# Run the Streamlit application in headless mode, optionally pinned to a CPU core
echo "Starting Streamlit on port $STREAMLIT_PORT..."
if [ -n "$CPU_CORE" ]; then
    echo "Pinning Streamlit to CPU core $CPU_CORE"
    taskset -c "$CPU_CORE" \
        streamlit run --server.headless True --server.fileWatcherType none \
        --server.address 0.0.0.0 --server.port $STREAMLIT_PORT autolabel_gui.py &
else
    streamlit run --server.headless True --server.fileWatcherType none \
        --server.address 0.0.0.0 --server.port $STREAMLIT_PORT autolabel_gui.py &
fi
STREAMLIT_PID=$!

# Wait for the Streamlit process to exit
wait $STREAMLIT_PID

#!/bin/bash

# Default virtual environment directory
DEFAULT_VENV_PATH="../envs/auto-label-engine"
VENV_PATH="$DEFAULT_VENV_PATH"

# Check if the virtual environment activation file exists
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

# Run the Streamlit application in headless mode
streamlit run --server.headless True --server.fileWatcherType none autolabel_gui.py

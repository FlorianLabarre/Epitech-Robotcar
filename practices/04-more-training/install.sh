#!/bin/bash

# Name of the virtual environment
venv_name=venv

# Create the virtual environment
python -m venv $venv_name

# Activate the virtual environment
source $venv_name/bin/activate

# Install required modules
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Deactivate the virtual environment
deactivate
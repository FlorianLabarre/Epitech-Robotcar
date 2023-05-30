@echo off

REM Name of the virtual environment
set venv_name=venv

REM Create the virtual environment
python -m venv %venv_name%

REM Activate the virtual environment
call %venv_name%\Scripts\activate.bat

REM Install required modules
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Deactivate the virtual environment
deactivate
#!/bin/bash
source .env
uvicorn main:app --host 0.0.0.0 --port 9999 --reload --reload-exclude client_ui.py

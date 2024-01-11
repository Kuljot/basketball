#!/bin/bash

echo "Starting application..."
src/ssl-proxy-linux-amd64 -from 0.0.0.0:443 -to 127.0.0.1:8501 &
streamlit run app.py



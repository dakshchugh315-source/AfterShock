#!/usr/bin/env bash
# Start-up for the Hugging Face Space: fetch the data bundle once, then run the dashboard.
set -e
python hf/fetch_data.py
exec streamlit run app/dashboard.py --server.port 7860 --server.address 0.0.0.0 --server.headless true

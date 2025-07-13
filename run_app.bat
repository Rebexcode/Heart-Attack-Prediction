@echo off
echo Starting Heart Attack Prediction App...
echo.
echo Installing required packages if needed...
pip install -r requirements.txt
echo.
echo Starting Streamlit application...
echo.
echo The app will open in your default browser at http://localhost:8501
echo Press Ctrl+C to stop the application
echo.
streamlit run app.py

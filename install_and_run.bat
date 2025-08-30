@echo off
echo Installing required Python packages...
pip install streamlit tensorflow Pillow numpy opencv-python

if %errorlevel% equ 0 (
    echo Installation completed successfully!
    echo Starting Streamlit app...
    streamlit run app.py
) else (
    echo Installation failed. Please check the error messages above.
    echo You can try installing packages manually with:
    echo pip install streamlit tensorflow Pillow numpy opencv-python
    pause
)

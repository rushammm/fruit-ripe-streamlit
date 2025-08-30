# Streamlit Fruit Classification App - Installation Script
# Run this script in PowerShell to install dependencies and start the app

Write-Host "Installing required Python packages..." -ForegroundColor Green

# Install packages
pip install streamlit tensorflow Pillow numpy opencv-python

if ($LASTEXITCODE -eq 0) {
    Write-Host "Installation completed successfully!" -ForegroundColor Green
    Write-Host "Starting Streamlit app..." -ForegroundColor Yellow
    
    # Start the Streamlit app
    streamlit run app.py
} else {
    Write-Host "Installation failed. Please check the error messages above." -ForegroundColor Red
    Write-Host "You can try installing packages manually with:" -ForegroundColor Yellow
    Write-Host "pip install streamlit tensorflow Pillow numpy opencv-python" -ForegroundColor White
}

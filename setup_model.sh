#!/bin/bash
# Download the Real-ESRGAN x4plus model weights

echo "Setting up Real-ESRGAN model..."

mkdir -p weights

if [ ! -f weights/RealESRGAN_x4plus.pth ]; then
    echo "Downloading RealESRGAN_x4plus.pth (64MB)..."
    curl -L -o weights/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
    echo "Download complete!"
else
    echo "Model weights already exist."
fi

echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""
echo "Setup complete! Run: uvicorn main:app --host 0.0.0.0 --port 8001 --reload"

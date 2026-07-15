#!/bin/bash
# Start the Image Enhancer web app

echo "Starting Image Enhancer..."
echo "Open http://localhost:8001 in your browser"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8001 --reload

#!/bin/bash
# Test script for light-object-detect API

set -e

API_URL="${API_URL:-http://localhost:9001}"
BACKEND="${BACKEND:-onnx}"
CONFIDENCE="${CONFIDENCE:-0.5}"

echo "=========================================="
echo "Light Object Detection API Test"
echo "=========================================="
echo "API URL: $API_URL"
echo "Backend: $BACKEND"
echo "Confidence: $CONFIDENCE"
echo ""

# Check if API is running
echo "1. Checking API health..."
if curl -s "$API_URL/" > /dev/null; then
    echo "✓ API is running"
    curl -s "$API_URL/" | python3 -m json.tool
else
    echo "✗ API is not responding"
    exit 1
fi

echo ""
echo "2. Checking backend status..."
curl -s "$API_URL/api/v1/backends" | python3 -m json.tool

echo ""
echo "3. Creating test image..."
python3 -c "
from PIL import Image, ImageDraw
import os

# Create a simple test image with shapes
img = Image.new('RGB', (640, 480), color='white')
draw = ImageDraw.Draw(img)

# Draw a person-like shape
draw.rectangle([200, 100, 400, 400], fill='blue', outline='black', width=3)
draw.ellipse([250, 50, 350, 150], fill='pink', outline='black', width=2)

# Draw a car-like shape
draw.rectangle([50, 300, 150, 400], fill='red', outline='black', width=3)
draw.rectangle([60, 320, 140, 380], fill='lightblue', outline='black', width=2)

img.save('test_image.jpg')
print('✓ Test image created: test_image.jpg')
"

echo ""
echo "4. Testing detection with $BACKEND backend..."
RESPONSE=$(curl -s -X POST "$API_URL/api/v1/detect?backend=$BACKEND&confidence_threshold=$CONFIDENCE&return_image=false" \
    -F "file=@test_image.jpg")

echo "$RESPONSE" | python3 -c "
import sys
import json

try:
    data = json.load(sys.stdin)
    
    print(f\"Backend: {data.get('backend', 'N/A')}\")
    print(f\"Filename: {data.get('filename', 'N/A')}\")
    print(f\"Image size: {data.get('image_width', 0)}x{data.get('image_height', 0)}\")
    print(f\"Process time: {data.get('process_time_ms', 0)}ms\")
    print(f\"Detections: {len(data.get('detections', []))}\")
    print()
    
    if data.get('detections'):
        print('Detected objects:')
        for i, det in enumerate(data['detections'], 1):
            print(f\"  {i}. {det['class_name']} (confidence: {det['confidence']:.2f})\")
            bbox = det['bbox']
            print(f\"     bbox: x={bbox['x']:.3f}, y={bbox['y']:.3f}, w={bbox['width']:.3f}, h={bbox['height']:.3f}\")
    else:
        print('No objects detected')
        
except json.JSONDecodeError as e:
    print(f'Error parsing JSON: {e}')
    print('Raw response:')
    print(sys.stdin.read())
except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "5. Testing with TFLite backend..."
RESPONSE=$(curl -s -X POST "$API_URL/api/v1/detect?backend=tflite&confidence_threshold=$CONFIDENCE&return_image=false" \
    -F "file=@test_image.jpg")

echo "$RESPONSE" | python3 -c "
import sys
import json

try:
    data = json.load(sys.stdin)
    print(f\"Backend: {data.get('backend', 'N/A')}\")
    print(f\"Process time: {data.get('process_time_ms', 0)}ms\")
    print(f\"Detections: {len(data.get('detections', []))}\")
    
    if data.get('detections'):
        print('Detected objects:')
        for i, det in enumerate(data['detections'], 1):
            print(f\"  {i}. {det['class_name']} (confidence: {det['confidence']:.2f})\")
    else:
        print('No objects detected')
        
except json.JSONDecodeError as e:
    print(f'Error parsing JSON: {e}')
except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="


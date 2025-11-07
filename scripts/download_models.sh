#!/bin/bash
# Script to download pre-trained ONNX models for object detection

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODEL_DIR="backends/onnx/models"
MODEL_SIZE="n"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --output-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-size SIZE    Model size: n, s, m, l, x (default: n)"
            echo "  --output-dir DIR     Output directory (default: backends/onnx/models)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Object Detection Model Download Tool"
echo "=========================================="
echo ""

# Create model directory
mkdir -p "$MODEL_DIR"

# Download COCO labels
echo -e "${YELLOW}Creating COCO labels file...${NC}"
cat > "$MODEL_DIR/coco_labels.txt" << 'EOF'
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
EOF

echo -e "${GREEN}✓ Labels file created: $MODEL_DIR/coco_labels.txt${NC}"

# Try to download YOLOv8 ONNX model from various sources
echo ""
echo -e "${YELLOW}Attempting to download YOLOv8${MODEL_SIZE} ONNX model...${NC}"

# Option 1: Try direct download from Hugging Face
ONNX_FILE="$MODEL_DIR/yolov8${MODEL_SIZE}.onnx"

echo "Trying Hugging Face repository..."
if wget -q --show-progress -O "$ONNX_FILE" \
    "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8${MODEL_SIZE}.onnx" 2>/dev/null; then
    echo -e "${GREEN}✓ Model downloaded successfully!${NC}"
else
    echo -e "${YELLOW}Hugging Face download failed, trying alternative...${NC}"
    
    # Option 2: Try GitHub releases
    if wget -q --show-progress -O "$ONNX_FILE" \
        "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8${MODEL_SIZE}.onnx" 2>/dev/null; then
        echo -e "${GREEN}✓ Model downloaded successfully!${NC}"
    else
        echo -e "${RED}✗ Automatic download failed.${NC}"
        echo ""
        echo -e "${YELLOW}Please use one of these methods:${NC}"
        echo ""
        echo "Method 1: Install ultralytics and export manually"
        echo "  pip install ultralytics"
        echo "  python3 scripts/download_models.py --model-size $MODEL_SIZE"
        echo ""
        echo "Method 2: Download from Hugging Face manually"
        echo "  Visit: https://huggingface.co/Ultralytics/YOLOv8/tree/main"
        echo "  Download: yolov8${MODEL_SIZE}.onnx"
        echo "  Save to: $MODEL_DIR/yolov8${MODEL_SIZE}.onnx"
        echo ""
        echo "Method 3: Use TFLite backend (already included)"
        echo "  Set BACKEND=tflite in your .env file"
        echo ""
        exit 1
    fi
fi

# Verify the downloaded file
if [ -f "$ONNX_FILE" ]; then
    FILE_SIZE=$(stat -f%z "$ONNX_FILE" 2>/dev/null || stat -c%s "$ONNX_FILE" 2>/dev/null)
    if [ "$FILE_SIZE" -lt 1000 ]; then
        echo -e "${RED}✗ Downloaded file is too small, likely an error page${NC}"
        rm -f "$ONNX_FILE"
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Setup Complete!${NC}"
    echo "=========================================="
    echo ""
    echo "Model: $ONNX_FILE"
    echo "Labels: $MODEL_DIR/coco_labels.txt"
    echo ""
    echo "Update your .env file with:"
    echo "  BACKEND=onnx"
    echo "  ONNX_MODEL_PATH=$ONNX_FILE"
    echo "  ONNX_LABELS_PATH=$MODEL_DIR/coco_labels.txt"
    echo "  ONNX_MODEL_TYPE=yolov8"
    echo ""
else
    echo -e "${RED}✗ Model file not found after download${NC}"
    exit 1
fi


#!/bin/bash
# Direct download script for YOLOv8 ONNX models
# This script downloads pre-converted ONNX models without requiring ultralytics

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
echo "YOLOv8 ONNX Model Download Tool"
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

# Try to download YOLOv8 ONNX model
echo ""
echo -e "${YELLOW}Downloading YOLOv8${MODEL_SIZE} ONNX model...${NC}"

ONNX_FILE="$MODEL_DIR/yolov8${MODEL_SIZE}.onnx"

# Try multiple sources
SOURCES=(
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8${MODEL_SIZE}.onnx"
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8${MODEL_SIZE}.onnx"
    "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8${MODEL_SIZE}.onnx"
    "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8${MODEL_SIZE}.onnx"
)

DOWNLOADED=false

for SOURCE in "${SOURCES[@]}"; do
    echo "Trying: $SOURCE"
    if wget -q --show-progress -O "$ONNX_FILE" "$SOURCE" 2>/dev/null; then
        # Check if file is valid (not an error page)
        FILE_SIZE=$(stat -f%z "$ONNX_FILE" 2>/dev/null || stat -c%s "$ONNX_FILE" 2>/dev/null)
        if [ "$FILE_SIZE" -gt 100000 ]; then
            echo -e "${GREEN}✓ Model downloaded successfully!${NC}"
            DOWNLOADED=true
            break
        else
            echo -e "${YELLOW}Downloaded file too small, trying next source...${NC}"
            rm -f "$ONNX_FILE"
        fi
    else
        echo -e "${YELLOW}Download failed, trying next source...${NC}"
    fi
done

if [ "$DOWNLOADED" = false ]; then
    echo -e "${RED}✗ All download attempts failed.${NC}"
    echo ""
    echo -e "${YELLOW}Alternative options:${NC}"
    echo ""
    echo "1. Use TFLite backend (already working):"
    echo "   bash scripts/download_tflite_models.sh --model-type ssd_mobilenet_v1"
    echo "   Then set BACKEND=tflite in your .env file"
    echo ""
    echo "2. Install ultralytics and export manually:"
    echo "   pip install ultralytics"
    echo "   python3 scripts/download_models.py --model-size $MODEL_SIZE"
    echo ""
    echo "3. Download manually from:"
    echo "   https://github.com/ultralytics/assets/releases"
    echo "   Save as: $ONNX_FILE"
    echo ""
    exit 1
fi

# Verify the downloaded file
if [ -f "$ONNX_FILE" ]; then
    FILE_SIZE=$(stat -f%z "$ONNX_FILE" 2>/dev/null || stat -c%s "$ONNX_FILE" 2>/dev/null)
    FILE_SIZE_MB=$((FILE_SIZE / 1024 / 1024))
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Setup Complete!${NC}"
    echo "=========================================="
    echo ""
    echo "Model: $ONNX_FILE (${FILE_SIZE_MB}MB)"
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


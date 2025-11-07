#!/bin/bash
# Script to download TFLite models for object detection

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL_DIR="backends/tflite/models"
MODEL_TYPE="ssd_mobilenet_v1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
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
            echo "  --model-type TYPE    Model type: ssd_mobilenet_v1, ssd_mobilenet_v2, efficientdet_lite0 (default: ssd_mobilenet_v1)"
            echo "  --output-dir DIR     Output directory (default: backends/tflite/models)"
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
echo "TFLite Model Download Tool"
echo "=========================================="
echo ""

# Create model directory
mkdir -p "$MODEL_DIR"

# Download COCO labels for TFLite
echo -e "${YELLOW}Creating COCO labels file for TFLite...${NC}"
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

# Download TFLite model
echo ""
echo -e "${YELLOW}Downloading TFLite model: ${MODEL_TYPE}...${NC}"

case $MODEL_TYPE in
    ssd_mobilenet_v1)
        MODEL_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
        MODEL_FILE="detect.tflite"
        ;;
    ssd_mobilenet_v2)
        MODEL_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_ssd_v2_coco.tgz"
        MODEL_FILE="detect.tflite"
        ;;
    efficientdet_lite0)
        MODEL_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_efficientdet_lite0_detection_metadata_1.tflite"
        MODEL_FILE="efficientdet_lite0.tflite"
        ;;
    *)
        echo -e "${RED}Unknown model type: $MODEL_TYPE${NC}"
        exit 1
        ;;
esac

# Download and extract
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo -e "${BLUE}Downloading from: $MODEL_URL${NC}"

if [[ $MODEL_URL == *.zip ]]; then
    wget -q --show-progress "$MODEL_URL" -O model.zip
    unzip -q model.zip
    # Find the .tflite file
    TFLITE_FILE=$(find . -name "*.tflite" | head -n 1)
    if [ -z "$TFLITE_FILE" ]; then
        echo -e "${RED}✗ No .tflite file found in archive${NC}"
        cd -
        rm -rf "$TEMP_DIR"
        exit 1
    fi
elif [[ $MODEL_URL == *.tgz ]] || [[ $MODEL_URL == *.tar.gz ]]; then
    wget -q --show-progress "$MODEL_URL" -O model.tgz
    tar -xzf model.tgz
    # Find the .tflite file
    TFLITE_FILE=$(find . -name "*.tflite" | head -n 1)
    if [ -z "$TFLITE_FILE" ]; then
        echo -e "${RED}✗ No .tflite file found in archive${NC}"
        cd -
        rm -rf "$TEMP_DIR"
        exit 1
    fi
elif [[ $MODEL_URL == *.tflite ]]; then
    wget -q --show-progress "$MODEL_URL" -O "$MODEL_FILE"
    TFLITE_FILE="$MODEL_FILE"
else
    echo -e "${RED}✗ Unsupported file format${NC}"
    cd -
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Get the original directory path
ORIG_DIR=$(pwd)
cd - > /dev/null

# Copy model to destination
OUTPUT_FILE="$MODEL_DIR/${MODEL_TYPE}.tflite"
cp "$TEMP_DIR/$TFLITE_FILE" "$OUTPUT_FILE"

# Clean up
rm -rf "$TEMP_DIR"

# Verify the downloaded file
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null)
    if [ "$FILE_SIZE" -lt 1000 ]; then
        echo -e "${RED}✗ Downloaded file is too small, likely an error${NC}"
        rm -f "$OUTPUT_FILE"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Model downloaded successfully!${NC}"
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Setup Complete!${NC}"
    echo "=========================================="
    echo ""
    echo "Model: $OUTPUT_FILE"
    echo "Labels: $MODEL_DIR/coco_labels.txt"
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo ""
    echo "To use this model, configure your light-object-detect API:"
    echo ""
    echo "  BACKEND=tflite"
    echo "  TFLITE_MODEL_PATH=$OUTPUT_FILE"
    echo "  TFLITE_LABELS_PATH=$MODEL_DIR/coco_labels.txt"
    echo "  TFLITE_CONFIDENCE_THRESHOLD=0.5"
    echo ""
    echo "Or use it via API query parameter:"
    echo "  curl -X POST 'http://localhost:9001/api/v1/detect?backend=tflite' -F 'file=@image.jpg'"
    echo ""
else
    echo -e "${RED}✗ Model file not found after download${NC}"
    exit 1
fi


#!/usr/bin/env python3
"""
Script to test the object detection API with a sample image.
"""

import os
import sys
import requests
import json
import argparse
from pathlib import Path


def test_detect_endpoint(image_path, url="http://localhost:9001/api/v1/detect", backend="tflite", confidence=0.5):
    """Test the detect endpoint with an image."""
    print(f"Testing detect endpoint with image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Prepare the request
    files = {
        'file': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')
    }
    
    params = {
        'backend': backend,
        'confidence_threshold': confidence
    }
    
    # Send the request
    try:
        response = requests.post(url, files=files, params=params)
        
        # Check if request was successful
        if response.status_code == 200:
            # Parse the response
            result = response.json()
            
            # Print the result
            print("\nDetection Results:")
            print(f"Backend: {result['backend']}")
            print(f"Image: {result['filename']} ({result['image_width']}x{result['image_height']})")
            print(f"Process time: {result['process_time_ms']} ms")
            print(f"Detections: {len(result['detections'])}")
            
            # Print each detection
            for i, detection in enumerate(result['detections']):
                print(f"\nDetection {i+1}:")
                print(f"  Label: {detection['label']}")
                print(f"  Confidence: {detection['confidence']:.2f}")
                print(f"  Bounding Box: [x_min={detection['bounding_box']['x_min']:.2f}, "
                      f"y_min={detection['bounding_box']['y_min']:.2f}, "
                      f"x_max={detection['bounding_box']['x_max']:.2f}, "
                      f"y_max={detection['bounding_box']['y_max']:.2f}]")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Close the file
        files['file'][1].close()


def test_backends_endpoint(url="http://localhost:9001/api/v1/backends"):
    """Test the backends endpoint."""
    print("Testing backends endpoint")
    
    # Send the request
    try:
        response = requests.get(url)
        
        # Check if request was successful
        if response.status_code == 200:
            # Parse the response
            result = response.json()
            
            # Print the result
            print("\nAvailable Backends:")
            print(f"Default backend: {result['default_backend']}")
            
            # Print each backend
            for backend, info in result['backends'].items():
                print(f"\nBackend: {backend}")
                print(f"  Status: {info['status']}")
                
                if info['status'] == 'available':
                    print(f"  Model path: {info['model_info']['model_path']}")
                    print(f"  Labels path: {info['model_info']['labels_path']}")
                    print(f"  Input shape: {info['model_info']['input_shape']}")
                    print(f"  Number of labels: {info['model_info']['num_labels']}")
                else:
                    print(f"  Error: {info['error']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the object detection API")
    parser.add_argument("--image", type=str, help="Path to the image to test")
    parser.add_argument("--url", type=str, default="http://localhost:9001", help="Base URL of the API")
    parser.add_argument("--backend", type=str, default="tflite", help="Backend to use for detection")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--test-backends", action="store_true", help="Test the backends endpoint")
    
    args = parser.parse_args()
    
    # Test backends endpoint
    if args.test_backends:
        test_backends_endpoint(f"{args.url}/api/v1/backends")
    
    # Test detect endpoint
    if args.image:
        test_detect_endpoint(args.image, f"{args.url}/api/v1/detect", args.backend, args.confidence)
    
    # If no arguments provided, show help
    if not args.image and not args.test_backends:
        parser.print_help()


if __name__ == "__main__":
    main()

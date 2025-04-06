#!/usr/bin/env python3
"""
Test script for the image return feature in the object detection API.
"""

import requests
import argparse
import base64
import os
from PIL import Image
import io
import json


def main():
    parser = argparse.ArgumentParser(description="Test the object detection API with image return")
    parser.add_argument("--url", default="http://localhost:8000/v1/detect", help="API endpoint URL")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--save-dir", default=".", help="Directory to save the returned image")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--backend", default="tflite", help="Detection backend")
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Prepare the request
    with open(args.image, "rb") as f:
        files = {"file": (os.path.basename(args.image), f, "image/jpeg")}
        
        # Test 1: Without image return (default behavior)
        print("\n=== Test 1: Without image return (default behavior) ===")
        params = {
            "backend": args.backend,
            "confidence_threshold": args.confidence,
            "return_image": False
        }
        
        print(f"Sending request to {args.url}")
        response = requests.post(args.url, files=files, params=params)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Detected {len(result['detections'])} objects")
            print(f"Process time: {result['process_time_ms']} ms")
            print("Image field in response:", "image" in result)
            
            # Print detection details
            for i, detection in enumerate(result["detections"]):
                print(f"  {i+1}. {detection['label']} ({detection['confidence']:.2f})")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    # Test 2: With image return
    print("\n=== Test 2: With image return ===")
    with open(args.image, "rb") as f:
        files = {"file": (os.path.basename(args.image), f, "image/jpeg")}
        params = {
            "backend": args.backend,
            "confidence_threshold": args.confidence,
            "return_image": True
        }
        
        print(f"Sending request to {args.url}")
        response = requests.post(args.url, files=files, params=params)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Detected {len(result['detections'])} objects")
            print(f"Process time: {result['process_time_ms']} ms")
            print("Image field in response:", "image" in result and result["image"] is not None)
            
            # Print detection details
            for i, detection in enumerate(result["detections"]):
                print(f"  {i+1}. {detection['label']} ({detection['confidence']:.2f})")
            
            # Save the returned image if available
            if "image" in result and result["image"] is not None:
                image_data = base64.b64decode(result["image"]["base64_data"])
                image_format = result["image"]["content_type"].split("/")[-1]
                
                # Create output filename
                basename = os.path.splitext(os.path.basename(args.image))[0]
                output_path = os.path.join(args.save_dir, f"{basename}_annotated.{image_format}")
                
                # Save image
                with open(output_path, "wb") as img_file:
                    img_file.write(image_data)
                
                print(f"Saved annotated image to: {output_path}")
                
                # Display image if running in an environment with display
                try:
                    image = Image.open(io.BytesIO(image_data))
                    image.show()
                except Exception as e:
                    print(f"Could not display image: {e}")
            else:
                print("No image data returned in the response")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


if __name__ == "__main__":
    main()

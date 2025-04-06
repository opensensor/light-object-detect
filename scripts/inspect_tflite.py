#!/usr/bin/env python3
"""
Script to inspect the tflite package.
"""

import sys
import importlib

def inspect_module(module_name):
    """Inspect a module and print its attributes."""
    try:
        module = importlib.import_module(module_name)
        print(f"Module: {module_name}")
        print(f"Dir: {dir(module)}")
        
        # Try to find any interpreter-like classes
        for attr in dir(module):
            if "interpret" in attr.lower():
                print(f"Found potential interpreter: {attr}")
                
        return module
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        return None

if __name__ == "__main__":
    # Try different tflite-related modules
    modules_to_try = [
        "tflite",
        "tensorflow.lite",
        "tflite_runtime",
        "tflite_runtime.interpreter"
    ]
    
    for module_name in modules_to_try:
        print(f"\nTrying to import {module_name}...")
        module = inspect_module(module_name)
        if module:
            print(f"Successfully imported {module_name}")
        else:
            print(f"Failed to import {module_name}")

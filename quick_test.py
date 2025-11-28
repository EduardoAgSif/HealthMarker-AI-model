"""
Quick Test Script - Fast model testing without menus
Perfect for quick validation during hackathon prep
"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model.har_model import HARModel

def quick_test():
    """Quick test with 3 common scenarios"""
    
    print("\n" + "="*60)
    print("QUICK MODEL TEST - 3 Scenarios")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    try:
        # Match the saved model: 23 input channels, 13 output classes
        model = HARModel(num_classes=13, input_channels=23)
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        model.eval()
        print("   ✓ Model loaded successfully!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # Test scenarios - update input to have 23 features instead of 6
    scenarios = [
        ("Normal Elderly Person", [1.0, 36.5, 72, 98, 0, 0.5] + [0.0] * 17),  # Pad to 23 features
        ("Heart Attack Risk", [0.8, 36.8, 110, 94, 0, 0.1] + [0.0] * 17),
        ("Fall Detected", [3.5, 36.5, 75, 98, 1, 0.0] + [0.0] * 17)
    ]
    
    print("\n2. Testing scenarios...\n")
    
    for name, values in scenarios:
        # Create 50-sample window with same values
        # Shape: (batch=1, channels=23, sequence=50)
        data = np.array([values] * 50, dtype=np.float32).T  # Transpose to (23, 50)
        tensor = torch.tensor(data).unsqueeze(0)  # Add batch dimension -> (1, 23, 50)
        
        # Predict
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            conf = probs[0][pred].item()
        
        # Print result
        print(f"   {name:25s} → Class {pred} ({conf*100:.1f}% confidence)")
    
    print("\n" + "="*60)
    print("✓ Quick test complete!")
    print("\nFor detailed testing, run: python test_model.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    quick_test()
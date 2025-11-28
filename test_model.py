"""
Comprehensive Model Testing Script
Tests your trained HAR model with simulated elderly health scenarios
"""
import torch
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model.har_model import HARModel

# Model path
MODEL_PATH = "model.pth"

# Health condition labels (update these based on your training)
HEALTH_CONDITIONS = {
    0: "Normal - Healthy",
    1: "Heart Attack Risk",
    2: "Respiratory Issue",
    3: "Low Blood Pressure",
    4: "High Blood Pressure",
    5: "Poor Sleep Quality",
    6: "Low Blood Oxygen",
    7: "Wandering/Lost"
}

def load_model():
    """Load the trained model"""
    print("=" * 60)
    print("Loading trained model...")
    try:
        model = HARModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        print("✓ Model loaded successfully!")
        print("=" * 60)
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nMake sure 'model.pth' exists in the project root.")
        print("If not, run: python src/model/train.py")
        sys.exit(1)

def generate_scenario_data(scenario_type, window_size=50, feature_count=6):
    """
    Generate simulated sensor data for different health scenarios
    
    Features:
    0: Acceleration magnitude (movement/falls)
    1: Temperature (fever detection)
    2: Heart rate
    3: Blood oxygen (SpO2)
    4: Fall detection binary
    5: Movement speed (wandering)
    """
    
    if scenario_type == "normal":
        # Normal elderly person - sitting/light activity
        accel = np.random.normal(1.0, 0.1, window_size)  # Normal gravity
        temp = np.random.normal(36.5, 0.2, window_size)  # Normal temp
        heart_rate = np.random.normal(72, 5, window_size)  # Normal HR
        spo2 = np.random.normal(98, 1, window_size)  # Normal oxygen
        fall = np.zeros(window_size)  # No fall
        speed = np.random.normal(0.5, 0.2, window_size)  # Light movement
        
    elif scenario_type == "heart_attack":
        # Heart attack indicators: elevated HR, chest pressure
        accel = np.random.normal(0.8, 0.2, window_size)  # Reduced movement
        temp = np.random.normal(36.8, 0.3, window_size)  # Slight elevation
        heart_rate = np.random.normal(110, 15, window_size)  # Elevated HR
        spo2 = np.random.normal(94, 2, window_size)  # Slightly low
        fall = np.zeros(window_size)
        speed = np.random.normal(0.1, 0.1, window_size)  # Minimal movement
        
    elif scenario_type == "respiratory":
        # Respiratory issue: low oxygen, increased HR
        accel = np.random.normal(1.0, 0.15, window_size)
        temp = np.random.normal(37.0, 0.3, window_size)  # Possible fever
        heart_rate = np.random.normal(95, 8, window_size)  # Elevated
        spo2 = np.random.normal(88, 3, window_size)  # Low oxygen!
        fall = np.zeros(window_size)
        speed = np.random.normal(0.3, 0.2, window_size)
        
    elif scenario_type == "low_bp":
        # Low blood pressure: dizziness, slow movement
        accel = np.random.normal(0.9, 0.2, window_size)  # Unsteady
        temp = np.random.normal(36.3, 0.2, window_size)  # Slightly low
        heart_rate = np.random.normal(58, 5, window_size)  # Low HR
        spo2 = np.random.normal(97, 1, window_size)
        fall = np.zeros(window_size)
        speed = np.random.normal(0.2, 0.15, window_size)  # Slow movement
        
    elif scenario_type == "high_bp":
        # High blood pressure: elevated vitals
        accel = np.random.normal(1.0, 0.1, window_size)
        temp = np.random.normal(36.7, 0.2, window_size)
        heart_rate = np.random.normal(95, 10, window_size)  # Elevated
        spo2 = np.random.normal(97, 1, window_size)
        fall = np.zeros(window_size)
        speed = np.random.normal(0.4, 0.2, window_size)
        
    elif scenario_type == "poor_sleep":
        # Poor sleep quality: irregular patterns
        accel = np.random.normal(1.5, 0.5, window_size)  # Restless
        temp = np.random.normal(36.4, 0.3, window_size)
        heart_rate = np.random.normal(68, 12, window_size)  # Irregular
        spo2 = np.random.normal(96, 2, window_size)
        fall = np.zeros(window_size)
        speed = np.random.normal(0.3, 0.3, window_size)  # Restless
        
    elif scenario_type == "low_oxygen":
        # Low blood oxygen
        accel = np.random.normal(0.9, 0.2, window_size)
        temp = np.random.normal(36.5, 0.2, window_size)
        heart_rate = np.random.normal(88, 8, window_size)  # Compensating
        spo2 = np.random.normal(85, 3, window_size)  # Critical low!
        fall = np.zeros(window_size)
        speed = np.random.normal(0.2, 0.15, window_size)
        
    elif scenario_type == "wandering":
        # Dementia wandering: continuous movement
        accel = np.random.normal(1.2, 0.3, window_size)  # Active
        temp = np.random.normal(36.5, 0.2, window_size)
        heart_rate = np.random.normal(82, 8, window_size)
        spo2 = np.random.normal(97, 1, window_size)
        fall = np.zeros(window_size)
        speed = np.random.normal(1.5, 0.5, window_size)  # Walking continuously
        
    elif scenario_type == "fall":
        # Fall detected: sudden acceleration spike then drop
        accel = np.random.normal(1.0, 0.1, window_size)
        # Simulate fall in middle of window
        fall_point = window_size // 2
        accel[fall_point-2:fall_point] = 3.5  # Sudden spike
        accel[fall_point:fall_point+3] = 0.2  # Sudden drop
        
        temp = np.random.normal(36.5, 0.2, window_size)
        heart_rate = np.random.normal(75, 5, window_size)
        spo2 = np.random.normal(98, 1, window_size)
        fall = np.zeros(window_size)
        fall[fall_point:] = 1  # Fall detected flag
        speed = np.random.normal(0.5, 0.2, window_size)
        speed[fall_point:] = 0  # No movement after fall
        
    else:
        raise ValueError(f"Unknown scenario: {scenario_type}")
    
    # Stack features together
    data = np.stack([accel, temp, heart_rate, spo2, fall, speed], axis=1)
    
    return data.astype(np.float32)

def predict_with_confidence(model, data):
    """
    Make prediction with confidence scores
    
    Args:
        model: Trained PyTorch model
        data: numpy array (window_size, features)
    
    Returns:
        predicted_class, confidence, all_probabilities
    """
    # Convert to tensor and add batch dimension
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].numpy()

def print_prediction_results(scenario_name, predicted_class, confidence, all_probs):
    """Pretty print prediction results"""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name.upper()}")
    print(f"{'='*60}")
    print(f"\n🎯 PREDICTION: {HEALTH_CONDITIONS.get(predicted_class, 'Unknown')}")
    print(f"📊 CONFIDENCE: {confidence*100:.1f}%")
    print(f"\n📈 All Class Probabilities:")
    print("-" * 60)
    
    # Sort by probability
    sorted_indices = np.argsort(all_probs)[::-1]
    
    for idx in sorted_indices:
        prob = all_probs[idx]
        if prob > 0.01:  # Only show if > 1%
            bar_length = int(prob * 40)
            bar = "█" * bar_length
            condition = HEALTH_CONDITIONS.get(idx, f"Class {idx}")
            print(f"{condition:25s} | {bar:40s} {prob*100:5.1f}%")
    
    print("-" * 60)

def test_all_scenarios(model):
    """Test model with all health scenarios"""
    print("\n" + "="*60)
    print("TESTING ALL HEALTH SCENARIOS")
    print("="*60)
    
    scenarios = [
        "normal",
        "heart_attack",
        "respiratory",
        "low_bp",
        "high_bp",
        "poor_sleep",
        "low_oxygen",
        "wandering",
        "fall"
    ]
    
    results = []
    
    for scenario in scenarios:
        # Generate scenario data
        data = generate_scenario_data(scenario)
        
        # Make prediction
        pred_class, confidence, all_probs = predict_with_confidence(model, data)
        
        # Print results
        print_prediction_results(scenario, pred_class, confidence, all_probs)
        
        # Store results
        results.append({
            'scenario': scenario,
            'predicted_class': pred_class,
            'predicted_condition': HEALTH_CONDITIONS.get(pred_class, 'Unknown'),
            'confidence': confidence
        })
        
        input("\nPress Enter to continue to next scenario...")
    
    return results

def test_single_scenario(model, scenario_name):
    """Test a single scenario"""
    print(f"\nTesting scenario: {scenario_name}")
    
    try:
        data = generate_scenario_data(scenario_name)
        pred_class, confidence, all_probs = predict_with_confidence(model, data)
        print_prediction_results(scenario_name, pred_class, confidence, all_probs)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nAvailable scenarios:")
        print("  - normal")
        print("  - heart_attack")
        print("  - respiratory")
        print("  - low_bp")
        print("  - high_bp")
        print("  - poor_sleep")
        print("  - low_oxygen")
        print("  - wandering")
        print("  - fall")

def test_custom_data(model):
    """Test with custom sensor values"""
    print("\n" + "="*60)
    print("CUSTOM DATA INPUT")
    print("="*60)
    print("\nEnter sensor values (or press Enter for defaults):\n")
    
    try:
        accel = float(input("Acceleration magnitude (0-3, default=1.0): ") or "1.0")
        temp = float(input("Temperature °C (35-39, default=36.5): ") or "36.5")
        hr = float(input("Heart rate bpm (40-150, default=72): ") or "72")
        spo2 = float(input("Blood oxygen % (80-100, default=98): ") or "98")
        fall = float(input("Fall detected (0 or 1, default=0): ") or "0")
        speed = float(input("Movement speed m/s (0-3, default=0.5): ") or "0.5")
        
        # Create window with repeated values
        window_size = 50
        data = np.array([
            [accel, temp, hr, spo2, fall, speed]
        ] * window_size, dtype=np.float32)
        
        # Make prediction
        pred_class, confidence, all_probs = predict_with_confidence(model, data)
        print_prediction_results("Custom Input", pred_class, confidence, all_probs)
        
    except ValueError as e:
        print(f"Invalid input: {e}")

def show_model_info(model):
    """Display model architecture information"""
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture: {model.__class__.__name__}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"\nNumber of Classes: {len(HEALTH_CONDITIONS)}")
    print("\nHealth Conditions:")
    for idx, condition in HEALTH_CONDITIONS.items():
        print(f"  {idx}: {condition}")

def main_menu():
    """Interactive testing menu"""
    model = load_model()
    
    while True:
        print("\n" + "="*60)
        print("ELDERLY HEALTH MONITORING - MODEL TESTING")
        print("="*60)
        print("\nChoose an option:")
        print("  1. Test all scenarios")
        print("  2. Test specific scenario")
        print("  3. Test with custom values")
        print("  4. Show model information")
        print("  5. Quick test (Normal scenario)")
        print("  6. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            test_all_scenarios(model)
        elif choice == "2":
            scenario = input("\nEnter scenario name: ").strip().lower()
            test_single_scenario(model, scenario)
        elif choice == "3":
            test_custom_data(model)
        elif choice == "4":
            show_model_info(model)
        elif choice == "5":
            data = generate_scenario_data("normal")
            pred_class, confidence, all_probs = predict_with_confidence(model, data)
            print_prediction_results("Normal (Quick Test)", pred_class, confidence, all_probs)
        elif choice == "6":
            print("\nExiting... Goodbye! 👋")
            break
        else:
            print("\n❌ Invalid choice. Please enter 1-6.")
        
        if choice in ["2", "3", "5"]:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   ELDERLY HEALTH MONITORING SYSTEM - MODEL TESTER        ║
    ║   Test your trained AI model without ESP32 hardware      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Error: Model file '{MODEL_PATH}' not found!")
        print("\nPlease train your model first:")
        print("  python src/model/train.py")
        sys.exit(1)
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
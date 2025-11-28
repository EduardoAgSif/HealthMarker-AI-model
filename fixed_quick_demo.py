"""
Fixed Quick Demo for Hackathon
"""
import torch
import numpy as np
import sys
import os

# Add src to path to import your model
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.har_model import HARModel

class WearableDemo:
    def __init__(self, model_path="model.pth"):
        print("Loading model...")
        try:
            self.model = HARModel(num_classes=13, input_channels=23)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
        # Define class meanings
        self.class_names = {
            0: "Normal Activity", 1: "Walking", 2: "Running", 
            3: "Falling", 4: "Heart Rate Spike", 5: "Low Oxygen",
            6: "High Stress", 7: "Sleeping", 8: "Sitting",
            9: "Standing", 10: "Exercise", 11: "Emergency", 
            12: "Abnormal Pattern"
        }
        
        self.emergency_classes = [3, 4, 5, 6, 11, 12]
    
    def simulate_sensor_data(self, scenario):
        """Generate realistic sensor data for demo scenarios"""
        scenarios = {
            "normal": [1.0, 36.5, 72, 98, 120, 0.5] + [0.1] * 17,
            "fall": [3.5, 36.5, 85, 95, 0, 0.0] + [0.8] * 17,
            "heart_issue": [0.8, 37.2, 140, 88, 0, 0.1] + [0.6] * 17,
            "low_oxygen": [1.2, 36.8, 95, 82, 0, 0.3] + [0.7] * 17
        }
        
        base_data = scenarios.get(scenario, scenarios["normal"])
        noise = np.random.normal(0, 0.1, 23)
        return (np.array(base_data) + noise).astype(np.float32)
    
    def predict_health_status(self, sensor_data):
        """Make prediction on sensor data"""
        sequence = np.array([sensor_data] * 50).T
        tensor_data = torch.tensor(sequence).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(tensor_data)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = probs[0][pred_class].item()
            
        return pred_class, confidence, probs.numpy()
    
    def generate_alert_message(self, prediction, confidence):
        """Generate appropriate alert message"""
        class_name = self.class_names.get(prediction, "Unknown")
        
        if prediction in self.emergency_classes:
            return f"🚨 EMERGENCY ALERT: {class_name} detected! Confidence: {confidence:.1%}\nImmediate attention required!"
        else:
            return f"✅ Normal: {class_name} detected. Confidence: {confidence:.1%}"
    
    def run_demo(self):
        """Run interactive demo"""
        print("\n" + "="*50)
        print("🤖 WEARABLE HEALTH MONITORING DEMO")
        print("="*50)
        
        scenarios = [
            ("Normal Walking", "normal"),
            ("Fall Detection", "fall"), 
            ("Heart Issue", "heart_issue"),
            ("Low Oxygen", "low_oxygen")
        ]
        
        for scenario_name, scenario_type in scenarios:
            print(f"\n📊 Testing: {scenario_name}")
            print("-" * 30)
            
            sensor_data = self.simulate_sensor_data(scenario_type)
            pred_class, confidence, all_probs = self.predict_health_status(sensor_data)
            
            message = self.generate_alert_message(pred_class, confidence)
            print(message)
            
            # Show top 3 predictions
            top3_idx = np.argsort(all_probs[0])[-3:][::-1]
            print("Top predictions:")
            for idx in top3_idx:
                print(f"  {self.class_names[idx]}: {all_probs[0][idx]:.1%}")

if __name__ == "__main__":
    print("Starting demo...")
    try:
        demo = WearableDemo()
        demo.run_demo()
        print("\n🎉 Demo completed successfully!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
"""
Simple WhatsApp Bot Demo (No Flask Required)
"""
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.model.har_model import HARModel

class SimpleHealthBot:
    def __init__(self, model_path="model.pth"):
        print("Loading AI model for health monitoring...")
        self.model = HARModel(num_classes=13, input_channels=23)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        
        self.class_names = {
            0: "Normal Activity", 1: "Walking", 2: "Running", 
            3: "Falling", 4: "Heart Rate Spike", 5: "Low Oxygen",
            6: "High Stress", 7: "Sleeping", 8: "Sitting",
            9: "Standing", 10: "Exercise", 11: "Emergency", 
            12: "Abnormal Pattern"
        }
        
        self.commands = {
            'help': 'Show available commands',
            'status': 'Check current health status', 
            'simulate fall': 'Test emergency fall detection',
            'simulate heart': 'Test heart issue detection',
            'vitals': 'Get current vitals'
        }
    
    def process_command(self, command):
        command = command.lower().strip()
        
        if command == 'help':
            return self.show_help()
        elif command == 'status':
            return self.get_status()
        elif command == 'simulate fall':
            return self.simulate_emergency('fall')
        elif command == 'simulate heart':
            return self.simulate_emergency('heart_issue')
        elif command == 'vitals':
            return self.get_vitals()
        else:
            return "🤖 Health Monitor Bot\nType 'help' for commands"
    
    def show_help(self):
        response = "🩺 *HEALTH BOT COMMANDS:*\n\n"
        for cmd, desc in self.commands.items():
            response += f"• {cmd}: {desc}\n"
        return response
    
    def get_status(self):
        return "📊 *CURRENT STATUS*\n• Activity: Normal\n• Heart Rate: 72 bpm\n• Oxygen: 98%\n• Alert: None"
    
    def simulate_emergency(self, scenario):
        sensor_data = self.simulate_sensor_data(scenario)
        pred_class, confidence, _ = self.predict_health_status(sensor_data)
        
        class_name = self.class_names.get(pred_class, "Unknown")
        
        if pred_class in [3, 4, 5, 6, 11, 12]:
            return f"🚨 EMERGENCY DETECTED!\n• Type: {class_name}\n• Confidence: {confidence:.1%}\n• Action: Alerting emergency contacts!"
        else:
            return f"✅ Status Normal\n• Activity: {class_name}\n• Confidence: {confidence:.1%}"
    
    def get_vitals(self):
        return "❤️ *VITAL SIGNS*\n• Heart Rate: 72 bpm\n• Oxygen: 98%\n• Temp: 36.5°C\n• Activity: Walking"
    
    def simulate_sensor_data(self, scenario):
        scenarios = {
            "normal": [1.0, 36.5, 72, 98, 120, 0.5] + [0.1] * 17,
            "fall": [3.5, 36.5, 85, 95, 0, 0.0] + [0.8] * 17,
            "heart_issue": [0.8, 37.2, 140, 88, 0, 0.1] + [0.6] * 17
        }
        base_data = scenarios.get(scenario, scenarios["normal"])
        noise = np.random.normal(0, 0.1, 23)
        return (np.array(base_data) + noise).astype(np.float32)
    
    def predict_health_status(self, sensor_data):
        sequence = np.array([sensor_data] * 50).T
        tensor_data = torch.tensor(sequence).unsqueeze((0))
        
        with torch.no_grad():
            output = self.model(tensor_data)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = probs[0][pred_class].item()
            
        return pred_class, confidence, probs.numpy()

def demo_chat():
    """Simulate a WhatsApp conversation"""
    bot = SimpleHealthBot()
    
    print("💬 SIMULATING WHATSAPP CHAT WITH HEALTH BOT")
    print("=" * 50)
    
    test_chat = [
        "Hello",
        "help", 
        "status",
        "simulate fall",
        "vitals"
    ]
    
    for message in test_chat:
        print(f"\n👤 You: {message}")
        response = bot.process_command(message)
        print(f"🤖 Bot: {response}")
        print("-" * 50)

if __name__ == "__main__":
    demo_chat()
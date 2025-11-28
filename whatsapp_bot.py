"""
WhatsApp Bot for Wearable Health Monitoring
Uses Twilio for WhatsApp integration
"""
from flask import Flask, request, jsonify
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import os
from quick_demo import WearableDemo

app = Flask(__name__)
demo_system = WearableDemo()

# Twilio configuration (you'll get these from Twilio dashboard)
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_AUTH_TOKEN = "your_auth_token"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

class WhatsAppHealthBot:
    def __init__(self):
        self.commands = {
            'help': 'Show available commands',
            'status': 'Check current health status', 
            'simulate [scenario]': 'Simulate health scenario (normal/fall/heart/oxygen)',
            'alerts': 'Check recent alerts',
            'vitals': 'Get current vitals summary'
        }
    
    def handle_message(self, user_message, user_number):
        """Process incoming WhatsApp messages"""
        message = user_message.lower().strip()
        
        if message == 'help':
            return self.show_help()
        elif message == 'status':
            return self.get_current_status()
        elif message.startswith('simulate'):
            return self.simulate_scenario(message, user_number)
        elif message == 'alerts':
            return self.get_alerts()
        elif message == 'vitals':
            return self.get_vitals()
        else:
            return "🤖 Health Monitor Bot\n" \
                   "Send 'help' for available commands\n" \
                   "I can check your status, simulate emergencies, and monitor your vitals!"
    
    def show_help(self):
        help_text = "🩺 *WEARABLE HEALTH BOT COMMANDS:*\n\n"
        for cmd, desc in self.commands.items():
            help_text += f"• *{cmd}*: {desc}\n"
        help_text += "\nExample: 'simulate fall' to test emergency response"
        return help_text
    
    def get_current_status(self):
        # Simulate getting current status from wearable
        sensor_data = demo_system.simulate_sensor_data("normal")
        pred_class, confidence, _ = demo_system.predict_health_status(sensor_data)
        
        return f"📊 *CURRENT HEALTH STATUS*\n" \
               f"Status: {demo_system.class_names[pred_class]}\n" \
               f"Confidence: {confidence:.1%}\n" \
               f"Vitals: Normal\n" \
               f"Alert Level: {'🟢 Low' if pred_class not in demo_system.emergency_classes else '🔴 High'}"
    
    def simulate_scenario(self, message, user_number):
        try:
            scenario = message.split(' ')[1]
            valid_scenarios = ['normal', 'fall', 'heart', 'oxygen']
            
            if scenario not in valid_scenarios:
                return f"❌ Invalid scenario. Use: {', '.join(valid_scenarios)}"
            
            scenario_map = {
                'normal': 'normal',
                'fall': 'fall', 
                'heart': 'heart_issue',
                'oxygen': 'low_oxygen'
            }
            
            sensor_data = demo_system.simulate_sensor_data(scenario_map[scenario])
            pred_class, confidence, _ = demo_system.predict_health_status(sensor_data)
            
            response = demo_system.generate_alert_message(pred_class, confidence)
            
            # If emergency, simulate calling for help
            if pred_class in demo_system.emergency_classes:
                response += f"\n\n📞 *AUTOMATED ACTION:* Emergency contact notified for {user_number}"
                
            return response
            
        except IndexError:
            return "❌ Please specify a scenario: 'simulate normal', 'simulate fall', etc."
    
    def get_alerts(self):
        return "📋 *RECENT ALERTS*\n" \
               "• No recent alerts\n" \
               "• Last check: All systems normal\n" \
               "• Device battery: 87%"
    
    def get_vitals(self):
        return "❤️ *CURRENT VITALS*\n" \
               "• Heart Rate: 72 bpm\n" \
               "• Oxygen: 98%\n" \
               "• Temperature: 36.5°C\n" \
               "• Activity: Walking\n" \
               "• Stress: Low"

# Initialize bot
health_bot = WhatsAppHealthBot()

@app.route("/whatsapp", methods=['POST'])
def whatsapp_webhook():
    """Webhook for Twilio WhatsApp messages"""
    incoming_msg = request.values.get('Body', '')
    user_number = request.values.get('From', '')
    
    print(f"Received message from {user_number}: {incoming_msg}")
    
    # Process message
    response_text = health_bot.handle_message(incoming_msg, user_number)
    
    # Create Twilio response
    twilio_response = MessagingResponse()
    twilio_response.message(response_text)
    
    return str(twilio_response)

@app.route("/")
def home():
    return "🤖 Wearable Health WhatsApp Bot is running!"

def run_demo_mode():
    """Run in demo mode without WhatsApp"""
    print("🚀 DEMO MODE - Testing WhatsApp commands locally")
    bot = WhatsAppHealthBot()
    
    test_messages = [
        "help",
        "status", 
        "simulate fall",
        "vitals",
        "alerts"
    ]
    
    for msg in test_messages:
        print(f"\n👤 User: {msg}")
        print(f"🤖 Bot: {bot.handle_message(msg, '+1234567890')}")
        print("-" * 50)

if __name__ == "__main__":
    # For demo without Twilio setup
    run_demo_mode()
    
    # For actual WhatsApp bot (uncomment when Twilio is setup):
    # app.run(host='0.0.0.0', port=5000, debug=True)
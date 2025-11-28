"""
Generate Model Testing Report
Creates a comprehensive HTML report of model performance
Perfect for presentations and documentation
"""
import torch
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model.har_model import HARModel

CONDITIONS = {
    0: "Normal - Healthy",
    1: "Heart Attack Risk",
    2: "Respiratory Issue",
    3: "Low Blood Pressure",
    4: "High Blood Pressure",
    5: "Poor Sleep Quality",
    6: "Low Blood Oxygen",
    7: "Wandering/Lost"
}

def generate_scenario_data(scenario_type, window_size=50):
    """Generate test data"""
    scenarios = {
        "normal": [1.0, 36.5, 72, 98, 0, 0.5],
        "heart_attack": [0.8, 36.8, 110, 94, 0, 0.1],
        "respiratory": [1.0, 37.0, 95, 88, 0, 0.3],
        "low_bp": [0.9, 36.3, 58, 97, 0, 0.2],
        "high_bp": [1.0, 36.7, 95, 97, 0, 0.4],
        "poor_sleep": [1.5, 36.4, 68, 96, 0, 0.3],
        "low_oxygen": [0.9, 36.5, 88, 85, 0, 0.2],
        "wandering": [1.2, 36.5, 82, 97, 0, 1.5]
    }
    
    base = scenarios.get(scenario_type, scenarios["normal"])
    data = []
    for _ in range(window_size):
        sample = [v + np.random.normal(0, v*0.05) for v in base]
        data.append(sample)
    
    return np.array(data, dtype=np.float32)

def test_model(model, scenario_name):
    """Test model and return results"""
    data = generate_scenario_data(scenario_name)
    tensor = torch.tensor(data).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(output, dim=1).item()
        conf = probs[pred].item()
    
    return {
        'scenario': scenario_name,
        'predicted_class': pred,
        'predicted_condition': CONDITIONS[pred],
        'confidence': conf,
        'all_probabilities': probs.numpy()
    }

def generate_html_report(results):
    """Generate beautiful HTML report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Testing Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .test-result {{
            background: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .scenario-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .prediction {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 15px 0;
        }}
        .pred-label {{
            font-weight: 600;
            color: #555;
        }}
        .pred-value {{
            font-weight: bold;
            font-size: 1.1em;
            color: #667eea;
        }}
        .confidence-high {{ color: #10b981; }}
        .confidence-medium {{ color: #f59e0b; }}
        .confidence-low {{ color: #ef4444; }}
        .prob-bar {{
            background: #e5e7eb;
            height: 25px;
            border-radius: 4px;
            margin: 8px 0;
            overflow: hidden;
            position: relative;
        }}
        .prob-fill {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-weight: 600;
            font-size: 0.9em;
        }}
        .summary {{
            background: #667eea;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .summary h2 {{
            margin-top: 0;
            color: white;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Model Testing Report</h1>
        <div class="subtitle">
            Elderly Health Monitoring System<br>
            Generated: {timestamp}
        </div>
        
        <div class="summary">
            <h2>📊 Test Summary</h2>
            <p><strong>Total Scenarios Tested:</strong> {len(results)}</p>
            <p><strong>Model:</strong> HAR Model (Human Activity Recognition)</p>
            <p><strong>Number of Classes:</strong> {len(CONDITIONS)}</p>
        </div>
"""
    
    # Add individual test results
    for result in results:
        scenario = result['scenario'].replace('_', ' ').title()
        pred = result['predicted_condition']
        conf = result['confidence']
        
        # Determine confidence class
        if conf > 0.8:
            conf_class = "confidence-high"
        elif conf > 0.5:
            conf_class = "confidence-medium"
        else:
            conf_class = "confidence-low"
        
        html += f"""
        <div class="test-result">
            <div class="scenario-name">📋 {scenario}</div>
            <div class="prediction">
                <span class="pred-label">Prediction:</span>
                <span class="pred-value">{pred}</span>
            </div>
            <div class="prediction">
                <span class="pred-label">Confidence:</span>
                <span class="pred-value {conf_class}">{conf*100:.1f}%</span>
            </div>
            
            <div style="margin-top: 15px;">
                <strong>All Probabilities:</strong>
"""
        
        # Add probability bars
        for class_id, prob in enumerate(result['all_probabilities']):
            if prob > 0.01:  # Only show if > 1%
                condition = CONDITIONS[class_id]
                html += f"""
                <div class="prob-bar">
                    <div class="prob-fill" style="width: {prob*100}%">
                        {condition}: {prob*100:.1f}%
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    # Add summary table
    html += """
        <h2 style="margin-top: 40px;">📈 Results Table</h2>
        <table>
            <tr>
                <th>Scenario</th>
                <th>Predicted Condition</th>
                <th>Confidence</th>
            </tr>
"""
    
    for result in results:
        scenario = result['scenario'].replace('_', ' ').title()
        pred = result['predicted_condition']
        conf = result['confidence']
        
        html += f"""
            <tr>
                <td>{scenario}</td>
                <td>{pred}</td>
                <td>{conf*100:.1f}%</td>
            </tr>
"""
    
    html += """
        </table>
        
        <div class="footer">
            <p>🏆 Hackathon Durania Project</p>
            <p>Elderly Health Monitoring System with AI</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html

def main():
    print("\n" + "="*60)
    print("GENERATING MODEL TESTING REPORT")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    try:
        model = HARModel()
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        model.eval()
        print("   ✓ Model loaded successfully!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # Test all scenarios
    print("\n2. Testing scenarios...")
    scenarios = [
        "normal",
        "heart_attack",
        "respiratory",
        "low_bp",
        "high_bp",
        "poor_sleep",
        "low_oxygen",
        "wandering"
    ]
    
    results = []
    for scenario in scenarios:
        print(f"   Testing: {scenario}...")
        result = test_model(model, scenario)
        results.append(result)
    
    print("   ✓ All scenarios tested!")
    
    # Generate report
    print("\n3. Generating HTML report...")
    html = generate_html_report(results)
    
    # Save report
    report_filename = f"model_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"   ✓ Report saved: {report_filename}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in results:
        scenario = result['scenario'].replace('_', ' ').title()
        pred = result['predicted_condition']
        conf = result['confidence']
        print(f"{scenario:25s} → {pred:25s} ({conf*100:.1f}%)")
    
    print("\n" + "="*60)
    print(f"✓ Report generated successfully!")
    print(f"  Open '{report_filename}' in your browser to view")
    print("="*60 + "\n")

if __name__ == "__main__":
    if not os.path.exists("model.pth"):
        print("\n❌ Error: model.pth not found!")
        print("Train your model first: python src/model/train.py")
        sys.exit(1)
    
    main()
"""
Visual Model Testing - Creates plots of predictions
Great for presentations and understanding model behavior
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model.har_model import HARModel

# Health conditions
CONDITIONS = {
    0: "Normal",
    1: "Heart Attack",
    2: "Respiratory",
    3: "Low BP",
    4: "High BP",
    5: "Poor Sleep",
    6: "Low O₂",
    7: "Wandering"
}

def generate_scenario(scenario_type, window_size=50):
    """Generate test data for scenarios"""
    
    scenarios = {
        "normal": [1.0, 36.5, 72, 98, 0, 0.5],
        "heart_attack": [0.8, 36.8, 110, 94, 0, 0.1],
        "respiratory": [1.0, 37.0, 95, 88, 0, 0.3],
        "low_bp": [0.9, 36.3, 58, 97, 0, 0.2],
        "high_bp": [1.0, 36.7, 95, 97, 0, 0.4],
        "low_oxygen": [0.9, 36.5, 88, 85, 0, 0.2],
        "wandering": [1.2, 36.5, 82, 97, 0, 1.5],
        "fall": [3.5, 36.5, 75, 98, 1, 0.0]
    }
    
    if scenario_type not in scenarios:
        scenario_type = "normal"
    
    base_values = scenarios[scenario_type]
    
    # Add some variation
    data = []
    for _ in range(window_size):
        sample = [v + np.random.normal(0, v*0.05) for v in base_values]
        data.append(sample)
    
    return np.array(data, dtype=np.float32)

def plot_sensor_data(data, scenario_name):
    """Plot the sensor data"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f'Sensor Data - {scenario_name}', fontsize=16, fontweight='bold')
    
    feature_names = [
        'Acceleration (g)', 
        'Temperature (°C)',
        'Heart Rate (bpm)',
        'Blood O₂ (%)',
        'Fall Detected',
        'Speed (m/s)'
    ]
    
    colors = ['#e74c3c', '#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#1abc9c']
    
    for idx, (ax, name, color) in enumerate(zip(axes.flat, feature_names, colors)):
        ax.plot(data[:, idx], color=color, linewidth=2)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(data))
    
    plt.tight_layout()
    return fig

def plot_predictions(all_probs, scenario_name, predicted_class):
    """Plot prediction probabilities as bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(CONDITIONS.keys())
    labels = [CONDITIONS[i] for i in classes]
    probs = [all_probs[i] * 100 for i in classes]
    
    # Color bars - highlight predicted class
    colors = ['#2ecc71' if i == predicted_class else '#3498db' for i in classes]
    
    bars = ax.bar(labels, probs, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        if height > 1:  # Only show if > 1%
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Predictions - {scenario_name}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(probs) * 1.15)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add prediction text
    pred_text = f"Prediction: {CONDITIONS[predicted_class]} ({probs[predicted_class]:.1f}%)"
    ax.text(0.5, 0.95, pred_text, 
            transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            ha='center', va='top')
    
    plt.tight_layout()
    return fig

def test_scenario_with_plots(model, scenario_name):
    """Test a scenario and create visualizations"""
    print(f"\nTesting: {scenario_name}")
    print("-" * 60)
    
    # Generate data
    data = generate_scenario(scenario_name)
    
    # Make prediction
    tensor = torch.tensor(data).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(output, dim=1).item()
        conf = probs[pred].item()
    
    print(f"Predicted: {CONDITIONS[pred]} ({conf*100:.1f}% confidence)")
    
    # Create plots
    fig1 = plot_sensor_data(data, scenario_name.title())
    fig2 = plot_predictions(probs.numpy(), scenario_name.title(), pred)
    
    return fig1, fig2

def main():
    print("\n" + "="*60)
    print("VISUAL MODEL TESTING")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    try:
        model = HARModel()
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        model.eval()
        print("✓ Model loaded successfully!\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Menu
    scenarios = [
        "normal",
        "heart_attack", 
        "respiratory",
        "low_bp",
        "high_bp",
        "low_oxygen",
        "wandering",
        "fall"
    ]
    
    print("Available scenarios:")
    for i, s in enumerate(scenarios, 1):
        print(f"  {i}. {s.replace('_', ' ').title()}")
    print(f"  {len(scenarios)+1}. Test all scenarios")
    print(f"  {len(scenarios)+2}. Exit")
    
    while True:
        try:
            choice = input("\nChoose scenario (number): ").strip()
            
            if not choice:
                continue
                
            choice_num = int(choice)
            
            if choice_num == len(scenarios) + 2:
                print("Exiting...")
                break
            elif choice_num == len(scenarios) + 1:
                # Test all
                print("\nTesting all scenarios...")
                for scenario in scenarios:
                    fig1, fig2 = test_scenario_with_plots(model, scenario)
                plt.show()
                break
            elif 1 <= choice_num <= len(scenarios):
                scenario = scenarios[choice_num - 1]
                fig1, fig2 = test_scenario_with_plots(model, scenario)
                plt.show()
            else:
                print("Invalid choice!")
                
        except ValueError:
            print("Please enter a number!")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         VISUAL MODEL TESTING - WITH PLOTS                ║
    ║   Test your model and see beautiful visualizations       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg backend
    except ImportError:
        print("\n⚠️  Warning: matplotlib not installed!")
        print("Install with: pip install matplotlib")
        sys.exit(1)
    
    if not os.path.exists("model.pth"):
        print("\n❌ Error: model.pth not found!")
        print("Train your model first: python src/model/train.py")
        sys.exit(1)
    
    main()
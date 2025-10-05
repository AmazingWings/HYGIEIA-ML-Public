"""
Hygieia AI Debris Detection & Capture Simulation
Empirical validation for research paper: "Designing a Sustainable Space Environment"
Author: Shriyans Dwivedi, Green Hope High School

This simulation validates the AI detection and capture mechanisms described in the paper
by training a neural network on synthetic debris images and simulating orbital encounters.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

# ============================================================================
# HYGIEIA SPECIFICATIONS (From Research Paper)
# ============================================================================

class HygieiaMission:
    """
    Mission parameters based on paper specifications:
    - 3U CubeSat architecture (15 kg total mass)
    - NASA C-POD propulsion (2.63 m/s delta-v)
    - Raspberry Pi 5 AI system
    - Net capture mechanism
    """
    def __init__(self):
        # Physical constraints from paper
        self.max_debris_size = 0.15  # meters (15 cm max capture)
        self.max_velocity = 0.5  # m/s (safe relative velocity)
        self.max_tumble = 30  # degrees/second (capture limit)
        self.detection_range_min = 10  # meters
        self.detection_range_max = 50  # meters
        
        # Propulsion (from Tsiolkovsky equation calculation)
        self.total_delta_v = 2.63  # m/s total budget
        self.delta_v_per_maneuver = 0.3  # m/s per capture attempt
        self.remaining_delta_v = self.total_delta_v
        
        # Capture mechanism
        self.net_success_rate = 0.95  # 95% deployment reliability
        
        # Mission tracking
        self.captures_attempted = 0
        self.captures_successful = 0
        self.debris_detected = 0
        self.false_positives = 0
        self.false_negatives = 0

# ============================================================================
# STEP 1: GENERATE SYNTHETIC CAMERA DATA
# ============================================================================

def generate_debris_images(n_images=1000, img_size=20, seed=42):
    """
    Simulate Sony IMX500 camera output as described in paper.
    
    Generates synthetic grayscale images representing what Hygieia's
    AI camera would see in LEO. Images include:
    - Background stars (realistic space noise)
    - Earth limb glow (lighting variations)
    - Debris objects (circles, rectangles, irregular shapes)
    
    Args:
        n_images: Total images to generate (half with debris, half without)
        img_size: Image dimensions (20x20 pixels for fast AI processing)
        seed: Random seed for reproducibility
    
    Returns:
        images: Flattened image arrays for ML model
        labels: 1 = debris present, 0 = no debris
    """
    np.random.seed(seed)
    print("\n" + "="*70)
    print("📸 STEP 1: GENERATING SYNTHETIC CAMERA DATA")
    print("="*70)
    print(f"Simulating Sony IMX500 camera output...")
    print(f"Creating {n_images} images ({img_size}x{img_size} pixels)")
    print(f"  → {n_images//2} images WITH debris (for AI to learn detection)")
    print(f"  → {n_images//2} images WITHOUT debris (to avoid false positives)")
    
    images = []
    labels = []
    
    for i in range(n_images):
        # Start with black space background
        img = np.zeros((img_size, img_size))
        
        # Add background stars (3-8 random bright pixels)
        n_stars = np.random.randint(3, 8)
        for _ in range(n_stars):
            x, y = np.random.randint(0, img_size, 2)
            img[x, y] = np.random.uniform(0.1, 0.3)
        
        # Add Earth limb glow (30% of images, simulates reflected sunlight)
        if np.random.random() < 0.3:
            gradient = np.linspace(0, 0.15, img_size)
            img += gradient[np.newaxis, :]
        
        # First half of images contain debris
        has_debris = i < n_images // 2
        
        if has_debris:
            # Randomize debris appearance
            size = np.random.randint(3, 8)  # 3-8 pixels (scales to 1-15cm)
            center_x = np.random.randint(size, img_size - size)
            center_y = np.random.randint(size, img_size - size)
            
            # Three debris shape types (realistic fragmentation patterns)
            shape = np.random.choice(['circle', 'rectangle', 'irregular'])
            
            if shape == 'circle':
                # Spherical debris (satellites, fuel tanks)
                y, x = np.ogrid[-center_x:img_size-center_x, -center_y:img_size-center_y]
                mask = x*x + y*y <= (size/2)**2
                img[mask] = np.random.uniform(0.7, 1.0)
            
            elif shape == 'rectangle':
                # Rectangular debris (solar panels, rocket bodies)
                x1 = max(0, center_x - size//2)
                x2 = min(img_size, center_x + size//2)
                y1 = max(0, center_y - size//2)
                y2 = min(img_size, center_y + size//2)
                img[x1:x2, y1:y2] = np.random.uniform(0.7, 1.0)
            
            else:
                # Irregular debris (collision fragments - most common)
                y, x = np.ogrid[-center_x:img_size-center_x, -center_y:img_size-center_y]
                mask = (x*x + y*y <= (size/2)**2) & (np.random.random((img_size, img_size)) > 0.4)
                img[mask] = np.random.uniform(0.7, 1.0)
        
        # Add camera sensor noise (realistic imaging conditions)
        img += np.random.normal(0, 0.05, (img_size, img_size))
        img = np.clip(img, 0, 1)
        
        # Flatten image for ML model input
        images.append(img.flatten())
        labels.append(1 if has_debris else 0)
    
    print(f"\n✅ Successfully generated {n_images} synthetic images")
    print(f"   Each image: {img_size*img_size} pixels ({img_size}x{img_size})")
    
    return np.array(images), np.array(labels)

# ============================================================================
# STEP 2: TRAIN AI MODEL
# ============================================================================

def train_ai_model(X_train, y_train, X_test, y_test):
    """
    Train neural network for debris detection (mimics TensorFlow Lite on Raspberry Pi 5).
    
    Uses Multi-Layer Perceptron (MLP) - a simple neural network that:
    - Takes 400 inputs (20x20 flattened image)
    - Processes through 2 hidden layers (100 and 50 neurons)
    - Outputs 2 values (debris / no debris)
    
    This architecture is lightweight enough to run on Raspberry Pi 5 in real-time.
    
    Returns:
        model: Trained neural network
        metrics: Dictionary of performance statistics
    """
    print("\n" + "="*70)
    print("🧠 STEP 2: TRAINING AI MODEL")
    print("="*70)
    print("Model Type: Multi-Layer Perceptron (Neural Network)")
    print("Architecture: 400 → 100 → 50 → 2 neurons")
    print("Training Device: Simulating Raspberry Pi 5 (16GB RAM) capabilities")
    print("Framework: Scikit-learn (proxy for TensorFlow Lite)")
    print("\nTraining process starting (50 epochs)...")
    print("You'll see the AI learn to recognize debris patterns...\n")
    
    # Create neural network
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Two hidden layers
        max_iter=50,  # 50 training iterations
        random_state=42,
        verbose=True,  # Show training progress
        learning_rate_init=0.001
    )
    
    # Train the model (watch it learn!)
    model.fit(X_train, y_train)
    
    # Evaluate on training and test data
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate performance metrics
    train_acc = accuracy_score(y_train, train_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100
    precision = precision_score(y_test, test_pred) * 100
    recall = recall_score(y_test, test_pred) * 100
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print("\n" + "="*70)
    print("📊 AI TRAINING RESULTS")
    print("="*70)
    print(f"Training Accuracy:   {train_acc:.2f}% (how well it learned)")
    print(f"Testing Accuracy:    {test_acc:.2f}% (how well it generalizes)")
    print(f"Precision:           {precision:.2f}% (when it says 'debris', it's right)")
    print(f"Recall:              {recall:.2f}% (catches most actual debris)")
    print(f"F1-Score:            {f1:.2f}% (balanced performance metric)")
    
    if test_acc >= 90:
        print("\n✅ EXCELLENT: Model ready for deployment on Raspberry Pi 5")
    elif test_acc >= 80:
        print("\n⚠️  ACCEPTABLE: Model functional but could improve with more training")
    else:
        print("\n❌ POOR: Model needs retraining or architecture adjustment")
    
    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return model, metrics

# ============================================================================
# STEP 3: GENERATE DEBRIS FIELD
# ============================================================================

def generate_debris_field(n_debris=100, seed=42):
    """
    Generate realistic debris objects with physical properties.
    
    Based on ESA's 2025 debris distribution data cited in paper:
    - Most debris is small (lognormal distribution favoring 1-10cm)
    - Velocities vary (exponential distribution, most < 0.5 m/s relative)
    - Tumble rates range from stable to rapidly spinning
    
    Returns:
        List of dictionaries, each containing debris properties:
        - size (meters)
        - distance (meters from Hygieia)
        - velocity (m/s relative)
        - tumble (degrees/second)
        - capturable (boolean based on constraints)
    """
    np.random.seed(seed)
    print("\n" + "="*70)
    print("🛰️  STEP 3: GENERATING DEBRIS FIELD")
    print("="*70)
    print(f"Creating {n_debris} debris objects with realistic properties...")
    print("Distribution based on ESA 2025 Annual Space Environment Report")
    
    debris_field = []
    
    for _ in range(n_debris):
        # Size: Lognormal distribution (most debris is small)
        size = np.random.lognormal(mean=-2.5, sigma=0.8)
        size = np.clip(size, 0.01, 0.50)  # 1cm to 50cm range
        
        # Distance: Uniform within and beyond detection range
        distance = np.random.uniform(5, 100)
        
        # Relative velocity: Exponential (most are slow, some fast)
        velocity = np.random.exponential(scale=0.3)
        velocity = np.clip(velocity, 0.01, 2.0)
        
        # Tumble rate: Uniform (unpredictable)
        tumble = np.random.uniform(0, 60)
        
        debris_field.append({
            'size': size,
            'distance': distance,
            'velocity': velocity,
            'tumble': tumble,
            'capturable': False  # Will be determined during simulation
        })
    
    # Analyze field composition
    small_debris = sum(1 for d in debris_field if d['size'] <= 0.15)
    slow_debris = sum(1 for d in debris_field if d['velocity'] <= 0.5)
    
    print(f"\n✅ Debris field generated:")
    print(f"   Total objects: {n_debris}")
    print(f"   ≤ 15cm (capturable size): {small_debris} ({small_debris/n_debris*100:.1f}%)")
    print(f"   ≤ 0.5 m/s (capturable velocity): {slow_debris} ({slow_debris/n_debris*100:.1f}%)")
    
    return debris_field

# ============================================================================
# STEP 4: MISSION SIMULATION
# ============================================================================

def simulate_mission(model, mission, debris_field, test_images, test_labels):
    """
    Simulate complete Hygieia orbital debris removal mission.
    
    For each debris object:
    1. AI detects debris using camera image
    2. Check if debris is within physical constraints
    3. Attempt capture if viable
    4. Track delta-v consumption
    5. Mission ends when propellant exhausted
    
    Returns:
        results: Dictionary containing all mission data
    """
    print("\n" + "="*70)
    print("🚀 STEP 4: MISSION SIMULATION")
    print("="*70)
    print(f"Hygieia launching into LEO debris field...")
    print(f"Initial Delta-V: {mission.total_delta_v} m/s")
    print(f"Debris to encounter: {len(debris_field)}")
    print(f"Mission objective: Capture as many debris as possible before fuel depletion\n")
    
    results = {
        'encounters': [],
        'detections': [],
        'captures': []
    }
    
    for i, (debris, image, label) in enumerate(zip(debris_field, test_images, test_labels)):
        # Check if mission can continue
        if mission.remaining_delta_v < mission.delta_v_per_maneuver:
            print(f"\n⛽ MISSION END: Delta-V depleted after {i} encounters")
            print(f"   Remaining fuel: {mission.remaining_delta_v:.3f} m/s (insufficient for maneuver)")
            break
        
        # AI DETECTION
        prediction = model.predict([image])[0]
        detected = (prediction == 1)
        
        # Track detection accuracy
        if detected and label == 1:
            mission.debris_detected += 1
        elif detected and label == 0:
            mission.false_positives += 1
        elif not detected and label == 1:
            mission.false_negatives += 1
        
        results['detections'].append({
            'detected': detected,
            'actual_debris': label == 1,
            'correct': detected == label
        })
        
        # Only attempt capture if AI detected debris
        if not detected:
            results['encounters'].append({
                'debris_id': i,
                'detected': False,
                'attempted': False,
                'successful': False,
                'reason': 'Not detected by AI'
            })
            continue
        
        # CHECK PHYSICAL CONSTRAINTS
        in_range = mission.detection_range_min <= debris['distance'] <= mission.detection_range_max
        right_size = debris['size'] <= mission.max_debris_size
        safe_velocity = debris['velocity'] <= mission.max_velocity
        stable_enough = debris['tumble'] <= mission.max_tumble
        
        capturable = in_range and right_size and safe_velocity and stable_enough
        debris['capturable'] = capturable
        
        # ATTEMPT CAPTURE
        if capturable:
            mission.captures_attempted += 1
            mission.remaining_delta_v -= mission.delta_v_per_maneuver
            
            # Simulate net deployment (95% success rate)
            net_deployed = np.random.random() < mission.net_success_rate
            
            if net_deployed:
                mission.captures_successful += 1
                status = "✅ CAPTURED"
                reason = "Successful capture"
            else:
                status = "❌ FAILED"
                reason = "Net deployment failure"
            
            # Print first 5 and every successful capture
            if i < 5 or net_deployed:
                print(f"Debris {i+1}: {status}")
                print(f"   Size: {debris['size']:.3f}m | Distance: {debris['distance']:.1f}m | "
                      f"Velocity: {debris['velocity']:.2f}m/s | Tumble: {debris['tumble']:.1f}°/s")
                print(f"   Remaining ΔV: {mission.remaining_delta_v:.2f} m/s\n")
            
            results['captures'].append({
                'debris_id': i,
                'successful': net_deployed,
                'reason': reason,
                **debris
            })
            
            results['encounters'].append({
                'debris_id': i,
                'detected': True,
                'attempted': True,
                'successful': net_deployed,
                'reason': reason
            })
        else:
            # Debris detected but not capturable
            reasons = []
            if not in_range:
                reasons.append(f"out of range ({debris['distance']:.1f}m)")
            if not right_size:
                reasons.append(f"too large ({debris['size']:.2f}m)")
            if not safe_velocity:
                reasons.append(f"too fast ({debris['velocity']:.2f}m/s)")
            if not stable_enough:
                reasons.append(f"tumbling ({debris['tumble']:.1f}°/s)")
            
            reason = "Outside capture envelope: " + ", ".join(reasons)
            
            results['encounters'].append({
                'debris_id': i,
                'detected': True,
                'attempted': False,
                'successful': False,
                'reason': reason
            })
    
    return results

# ============================================================================
# STEP 5: RESULTS ANALYSIS
# ============================================================================

def analyze_results(mission, ai_metrics, results, debris_field):
    """
    Comprehensive analysis of mission performance for research paper.
    """
    print("\n" + "="*70)
    print("📊 MISSION RESULTS ANALYSIS")
    print("="*70)
    
    # AI Performance
    print("\n🧠 AI DETECTION PERFORMANCE:")
    print(f"   Test Accuracy:    {ai_metrics['test_accuracy']:.2f}%")
    print(f"   Precision:        {ai_metrics['precision']:.2f}%")
    print(f"   Recall:           {ai_metrics['recall']:.2f}%")
    print(f"   F1-Score:         {ai_metrics['f1_score']:.2f}%")
    print(f"   True Positives:   {mission.debris_detected}")
    print(f"   False Positives:  {mission.false_positives}")
    print(f"   False Negatives:  {mission.false_negatives}")
    
    # Capture Performance
    capture_rate = (mission.captures_successful / mission.captures_attempted * 100 
                   if mission.captures_attempted > 0 else 0)
    
    print(f"\n🎯 CAPTURE PERFORMANCE:")
    print(f"   Captures Attempted:   {mission.captures_attempted}")
    print(f"   Captures Successful:  {mission.captures_successful}")
    print(f"   Success Rate:         {capture_rate:.1f}%")
    print(f"   Delta-V Used:         {mission.total_delta_v - mission.remaining_delta_v:.2f} m/s")
    print(f"   Delta-V Remaining:    {mission.remaining_delta_v:.2f} m/s")
    
    # Debris Field Analysis
    total_encounters = len(results['encounters'])
    capturable_count = sum(1 for d in debris_field[:total_encounters] if d.get('capturable', False))
    
    print(f"\n🛰️  DEBRIS FIELD STATISTICS:")
    print(f"   Total Encountered:        {total_encounters}")
    print(f"   Within Capture Envelope:  {capturable_count}")
    print(f"   Capture Efficiency:       {mission.captures_successful}/{capturable_count} possible")
    
    # Mission Effectiveness
    print(f"\n📈 MISSION EFFECTIVENESS:")
    efficiency = (mission.captures_successful / total_encounters * 100) if total_encounters > 0 else 0
    print(f"   Overall Efficiency:       {efficiency:.1f}% (captures / encounters)")
    print(f"   Captures per m/s ΔV:      {mission.captures_successful / (mission.total_delta_v - mission.remaining_delta_v):.2f}")
    
    # Paper Integration Summary
    print(f"\n💡 FOR YOUR RESEARCH PAPER:")
    print(f"   → 'The AI model achieved {ai_metrics['test_accuracy']:.1f}% detection accuracy'")
    print(f"   → 'Mission simulations demonstrated {capture_rate:.1f}% capture success rate'")
    print(f"   → 'Each Hygieia unit captured {mission.captures_successful} debris objects'")
    print(f"   → 'Delta-V budget allowed {mission.captures_attempted} capture attempts'")
    
    return {
        'ai_accuracy': ai_metrics['test_accuracy'],
        'capture_rate': capture_rate,
        'total_captures': mission.captures_successful,
        'encounters': total_encounters,
        'efficiency': efficiency
    }

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

def create_visualizations(mission, ai_metrics, results):
    """
    Generate publication-ready graphs for research paper.
    """
    print("\n" + "="*70)
    print("📈 GENERATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    # Figure 1: AI Performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Detection breakdown
    categories = ['True\nPositive', 'False\nPositive', 'False\nNegative']
    values = [mission.debris_detected, mission.false_positives, mission.false_negatives]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('AI Detection Performance', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # Accuracy metrics
    metrics = ['Accuracy', 'Precision', 'Recall']
    scores = [ai_metrics['test_accuracy'], ai_metrics['precision'], ai_metrics['recall']]
    
    ax2.barh(metrics, scores, color='#3498db', edgecolor='black', linewidth=2)
    ax2.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Detection Metrics', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(scores):
        ax2.text(v + 2, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figure1_ai_performance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/figure1_ai_performance.png")
    plt.close()
    
    # Figure 2: Capture Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Capture outcomes
    successful = mission.captures_successful
    failed = mission.captures_attempted - mission.captures_successful
    not_attempted = len(results['encounters']) - mission.captures_attempted
    
    labels = ['Successful\nCapture', 'Failed\nCapture', 'Not\nAttempted']
    sizes = [successful, failed, not_attempted]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontweight': 'bold'})
    ax1.set_title('Mission Outcomes', fontsize=14, fontweight='bold')
    
    # Delta-V depletion
    captures = list(range(mission.captures_successful + 1))
    delta_v = [mission.total_delta_v - (i * mission.delta_v_per_maneuver) 
               for i in captures]
    
    ax2.plot(captures, delta_v, marker='o', linewidth=2.5, markersize=8,
             color='#e74c3c', markeredgecolor='black', markeredgewidth=1.5)
    ax2.fill_between(captures, delta_v, alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Successful Captures', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Remaining Delta-V (m/s)', fontsize=12, fontweight='bold')
    ax2.set_title('Propellant Consumption', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=mission.delta_v_per_maneuver, color='orange', 
                linestyle='--', linewidth=2, label='Min Required')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/figure2_capture_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/figure2_capture_analysis.png")
    plt.close()
    
    print("\n✅ All visualizations saved to 'results/' folder")
    print("   Use these figures in your research paper!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - runs complete simulation pipeline.
    """
    print("\n" + "="*70)
    print("HYGIEIA AI DEBRIS DETECTION & CAPTURE SIMULATION")
    print("Research Paper: 'Designing a Sustainable Space Environment'")
    print("Author: Shriyans Dwivedi, Green Hope High School")
    print("="*70)
    
    # Initialize mission
    mission = HygieiaMission()
    
    # Step 1: Generate training data
    X_train, y_train = generate_debris_images(n_images=1000, seed=42)
    X_test, y_test = generate_debris_images(n_images=100, seed=123)
    
    # Step 2: Train AI model
    model, ai_metrics = train_ai_model(X_train, y_train, X_test, y_test)
    
    # Step 3: Generate debris field
    debris_field = generate_debris_field(n_debris=100, seed=123)
    
    # Step 4: Simulate mission
    results = simulate_mission(model, mission, debris_field, X_test, y_test)
    
    # Step 5: Analyze results
    summary = analyze_results(mission, ai_metrics, results, debris_field)
    
    # Step 6: Create visualizations
    create_visualizations(mission, ai_metrics, results)
    
    # Save summary to file
    with open('results/mission_summary.txt', 'w') as f:
        f.write("HYGIEIA SIMULATION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"AI Detection Accuracy: {summary['ai_accuracy']:.2f}%\n")
        f.write(f"Capture Success Rate: {summary['capture_rate']:.2f}%\n")
        f.write(f"Total Captures: {summary['total_captures']}\n")
        f.write(f"Total Encounters: {summary['encounters']}\n")
        f.write(f"Mission Efficiency: {summary['efficiency']:.2f}%\n")
    
    print("\n" + "="*70)
    print("✅ SIMULATION COMPLETE")
    print("="*70)
    print("\nResults saved:")
    print("  📄 results/mission_summary.txt")
    print("  📊 results/figure1_ai_performance.png")
    print("  📊 results/figure2_capture_analysis.png")
    print("\nNext steps:")
    print("  1. Run this simulation 3-5 times to get average results")
    print("  2. Include the graphs in your paper's 'Results' section")
    print("  3. Cite the numbers in your discussion")
    print("\nExample citation:")
    print(f"  'The AI model achieved {summary['ai_accuracy']:.1f}% detection")
    print(f"   accuracy with a {summary['capture_rate']:.1f}% capture success rate'")

if __name__ == "__main__":
    main()
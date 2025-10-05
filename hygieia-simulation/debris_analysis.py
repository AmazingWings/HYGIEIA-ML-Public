"""
Enhanced Hygieia Debris Analysis
Author: Shriyans Dwivedi
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

def enhance_features(df):
    """Create sophisticated features based on ESA Space Debris Mitigation Requirements"""
    
    # Basic size conversion with ESA size classifications
    size_map = {'small': 0.05, 'medium': 0.5, 'large': 2.0}
    df['size_m'] = df['size_category'].map(size_map)
    
    # Constants
    G = 6.67430e-11  # gravitational constant
    M = 5.972e24     # Earth's mass
    R = 6371000      # Earth's radius in meters
    LEO_LIMIT = 2000000  # LEO limit in meters (2000 km)
    GEO_ALTITUDE = 35786000  # GEO altitude in meters
    
    # Convert altitude to meters
    df['altitude_m'] = df['altitude_km'] * 1000
    
    # Orbital zone classification (ESA zones)
    df['orbit_zone'] = pd.cut(df['altitude_m'], 
                             bins=[-np.inf, 200000, LEO_LIMIT, GEO_ALTITUDE-1000000, 
                                  GEO_ALTITUDE+1000000, np.inf],
                             labels=['Sub-LEO', 'LEO', 'MEO', 'GEO-approach', 'Beyond-GEO'])
    
    # Calculate orbital parameters
    df['orbital_velocity'] = np.sqrt(G * M / (R + df['altitude_m']))
    df['escape_velocity'] = np.sqrt(2 * G * M / (R + df['altitude_m']))
    df['orbital_energy'] = -G * M / (2 * (R + df['altitude_m']))
    
    # Enhanced orbital characteristics
    df['period_seconds'] = df['orbital_period_hours'] * 3600
    df['angular_velocity'] = 2 * np.pi / df['period_seconds']
    df['relative_velocity_potential'] = df['orbital_velocity'] / df['escape_velocity']
    
    # ESA-specific risk metrics
    df['collision_risk'] = df['orbital_velocity'] * df['size_m'] / df['altitude_m']
    df['debris_persistence'] = np.where(df['altitude_m'] < LEO_LIMIT,
                                      df['altitude_m'] / 200000,  # LEO decay factor
                                      1.0)  # Higher orbits persist longer
    
    # Maneuverability potential (based on size and velocity)
    df['maneuver_difficulty'] = df['size_m'] * df['orbital_velocity'] / 1000
    
    # Deorbit feasibility (ESA requirement consideration)
    df['natural_decay_years'] = np.where(
        df['altitude_m'] < 500000,  # Below 500km
        (df['altitude_m'] / 100000) ** 2,  # Approximate decay time
        999  # Long-term persistence
    )
    
    # Area-to-mass ratio estimation (important for decay calculations)
    df['area_to_mass'] = (np.pi * (df['size_m']/2)**2) / df['mass_kg']
    
    # Normalized risk score (0-1)
    df['risk_score'] = (
        (df['collision_risk'] / df['collision_risk'].max()) * 0.4 +
        (df['debris_persistence']) * 0.3 +
        (df['maneuver_difficulty'] / df['maneuver_difficulty'].max()) * 0.3
    )
    
    return df
    
    # Calculate orbital energy
    df['orbital_energy'] = -G * M / (2 * (R + df['altitude_m']))
    
    # Calculate orbital period in seconds
    df['period_seconds'] = df['orbital_period_hours'] * 3600
    
    # Calculate angular velocity (rad/s)
    df['angular_velocity'] = 2 * np.pi / df['period_seconds']
    
    # Calculate relative velocity potential (normalized)
    df['relative_velocity_potential'] = df['orbital_velocity'] / df['escape_velocity']
    
    return df

def determine_capturability(df):
    """Enhanced debris capturability assessment based on ESA requirements"""
    
    # 1. Physical Constraints
    # Size constraint (≤ 15cm as per Hygieia capabilities)
    size_ok = df['size_m'] <= 0.15
    
    # Mass-to-size ratio check (density consideration)
    density_ok = df['mass_kg'] / (4/3 * np.pi * (df['size_m']/2)**3) < 8000  # Max density check
    
    # 2. Orbital Parameters
    # Focus on LEO objects (highest priority per ESA)
    leo_priority = df['orbit_zone'] == 'LEO'
    
    # Velocity constraints (based on safe approach capabilities)
    velocity_ok = df['orbital_velocity'] <= (df['orbital_velocity'].mean() + 0.5)
    
    # 3. Risk Assessment
    # Lower risk scores are preferred
    risk_ok = df['risk_score'] < 0.7  # Threshold for acceptable risk
    
    # Natural decay consideration (prefer objects that won't decay naturally soon)
    decay_priority = df['natural_decay_years'] > 5  # Focus on persistent debris
    
    # 4. Operational Feasibility
    # Altitude within operational range
    altitude_ok = (df['altitude_km'] >= 200) & (df['altitude_km'] <= 2000)
    
    # Energy constraints for efficient capture
    energy_ok = abs(df['orbital_energy']) <= abs(df['orbital_energy'].mean() * 1.1)
    
    # Maneuverability check
    maneuver_ok = df['maneuver_difficulty'] < df['maneuver_difficulty'].quantile(0.7)
    
    # 5. ESA Compliance Scoring
    # Calculate compliance score based on multiple factors
    df['esa_compliance_score'] = (
        (size_ok.astype(int) * 0.3) +
        (leo_priority.astype(int) * 0.2) +
        (risk_ok.astype(int) * 0.2) +
        (decay_priority.astype(int) * 0.15) +
        (maneuver_ok.astype(int) * 0.15)
    )
    
    # 6. Final Capturability Assessment
    # Combine all criteria with ESA compliance
    df['capturable'] = (
        size_ok &
        density_ok &
        leo_priority &
        velocity_ok &
        risk_ok &
        altitude_ok &
        energy_ok &
        maneuver_ok &
        (df['esa_compliance_score'] >= 0.7)  # Minimum compliance threshold
    ).astype(int)
    
    return df

def analyze_debris(data_path='space_debris_data.csv'):
    """Advanced debris analysis with enhanced ML"""
    
    # Load and prepare data
    print("\nLoading and enhancing debris data...")
    df = pd.read_csv(data_path)
    df = enhance_features(df)
    df = determine_capturability(df)
    
    capturable_count = df['capturable'].sum()
    print(f"Analyzed {len(df)} objects")
    print(f"Identified {capturable_count} capturable objects ({capturable_count/len(df)*100:.1f}%)")
    
    # Prepare features for ML
    features = [
        'altitude_km', 'mass_kg', 'orbital_period_hours', 'inclination_deg',
        'orbital_velocity', 'escape_velocity', 'orbital_energy',
        'angular_velocity', 'relative_velocity_potential'
    ]
    
    X = df[features]
    y = df['capturable']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Setup model with enhanced parameters
    print("\nTraining enhanced ML model...")
    base_model = RandomForestClassifier(
        n_estimators=500,          # Increased number of trees
        max_depth=15,             # Slightly deeper trees
        min_samples_split=5,      
        min_samples_leaf=2,       # Prevent overfitting
        max_features='sqrt',      # Optimal for RF
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    
    # Setup cross-validation with stratification
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform grid search for hyperparameter tuning
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [400, 500, 600],
        'max_depth': [10, 15, 20],
        'min_samples_split': [4, 5, 6],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Performing hyperparameter optimization...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=skf,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"\nBest parameters found: {grid_search.best_params_}")
    
    # Get the best model
    model = grid_search.best_estimator_
    
    # Perform cross-validation with the best model
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='balanced_accuracy')
    print(f"\nCross-validation balanced accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*2*100:.1f}%)")
    
    # Final training and prediction
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate prediction probabilities for confidence analysis
    y_pred_proba = model.predict_proba(X_test)
    confidence_scores = np.max(y_pred_proba, axis=1)
    high_confidence = (confidence_scores > 0.9).sum() / len(confidence_scores)
    
    # Comprehensive model evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
    
    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    balanced_acc = balanced_accuracy_score(y_test, y_pred) * 100
    
    # Handle potential zero division in precision/recall
    precision = precision_score(y_test, y_pred, zero_division=0) * 100
    recall = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    
    print(f"\nFinal Model Performance:")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Balanced Accuracy: {balanced_acc:.1f}%")
    print(f"Precision: {precision:.1f}%")
    print(f"Recall: {recall:.1f}%")
    print(f"F1-Score: {f1:.1f}%")
    print(f"High Confidence Predictions: {high_confidence*100:.1f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate visualizations
    print("\nGenerating enhanced visualizations...")
    os.makedirs('results', exist_ok=True)
    
    # Prepare data for visualizations
    capturable = df[df['capturable'] == 1]
    non_capturable = df[df['capturable'] == 0]
    
    # 1. Advanced Feature Importance
    plt.figure(figsize=(12, 6))
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    sns.barplot(data=importance, x='importance', y='feature')
    plt.title('Orbital Parameters Influencing Debris Capture', fontsize=12, fontweight='bold')
    plt.xlabel('Relative Importance', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300)
    plt.close()
    
    # 2. Summary Statistics Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Debris Capture Summary (Bar Chart)
    plt.subplot(2, 2, 1)
    capture_summary = pd.DataFrame({
        'Status': ['Total Debris', 'Capturable'],
        'Count': [len(df), capturable_count]
    })
    sns.barplot(data=capture_summary, x='Status', y='Count', palette=['gray', 'green'])
    plt.title('Debris Capture Potential', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Objects', fontsize=10)
    for i, v in enumerate(capture_summary['Count']):
        plt.text(i, v, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                horizontalalignment='center', verticalalignment='bottom')
    
    # Plot 2: Altitude vs Velocity
    plt.subplot(2, 2, 2)
    plt.scatter(non_capturable['altitude_km'], non_capturable['orbital_velocity'],
               alpha=0.1, c='red', label='Non-Capturable', s=20)
    plt.scatter(capturable['altitude_km'], capturable['orbital_velocity'],
               alpha=0.6, c='green', label='Capturable', s=50)
    plt.xlabel('Altitude (km)', fontsize=10)
    plt.ylabel('Orbital Velocity (m/s)', fontsize=10)
    plt.title('Debris Distribution: Altitude vs Velocity', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Size Distribution
    plt.subplot(2, 2, 3)
    size_data = pd.DataFrame({
        'Category': ['Small (≤15cm)', 'Medium', 'Large'],
        'Count': [
            len(df[df['size_m'] <= 0.15]),
            len(df[(df['size_m'] > 0.15) & (df['size_m'] <= 0.5)]),
            len(df[df['size_m'] > 0.5])
        ]
    })
    sns.barplot(data=size_data, x='Category', y='Count', palette='viridis')
    plt.title('Debris Size Distribution', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Objects', fontsize=10)
    plt.xticks(rotation=45)
    for i, v in enumerate(size_data['Count']):
        plt.text(i, v, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                horizontalalignment='center', verticalalignment='bottom')
    
    # Plot 4: Energy vs Mass with Capture Zones
    plt.subplot(2, 2, 4)
    plt.scatter(non_capturable['orbital_energy'], non_capturable['mass_kg'],
               alpha=0.1, c='red', label='Non-Capturable', s=20)
    plt.scatter(capturable['orbital_energy'], capturable['mass_kg'],
               alpha=0.6, c='green', label='Capturable', s=50)
    plt.xlabel('Orbital Energy (J)', fontsize=10)
    plt.ylabel('Mass (kg)', fontsize=10)
    plt.yscale('log')
    plt.title('Debris Distribution: Energy vs Mass', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/debris_distribution.png', dpi=300)
    plt.close()
    
    # Save enhanced analysis
    print("\nSaving comprehensive analysis...")
    with open('results/debris_analysis.txt', 'w') as f:
        f.write("HYGIEIA ADVANCED DEBRIS CAPTURE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ANALYSIS SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total debris analyzed: {len(df):,}\n")
        f.write(f"Capturable debris: {capturable_count:,} ({capturable_count/len(df)*100:.1f}%)\n\n")
        
        f.write("CAPTURE CONSTRAINTS\n")
        f.write("-" * 20 + "\n")
        f.write("- Maximum size: 15 cm\n")
        f.write("- Altitude range: 200-2000 km\n")
        f.write("- Relative velocity: ≤ 0.5 m/s\n")
        f.write("- Orbital energy: Within 10% of mean\n\n")
        
        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        f.write(f"- Cross-validation accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*2*100:.1f}%)\n")
        f.write(f"- Final accuracy: {accuracy:.1f}%\n\n")
        
        f.write("KEY ORBITAL PARAMETERS\n")
        f.write("-" * 20 + "\n")
        for _, row in importance.iterrows():
            f.write(f"- {row['feature']}: {row['importance']:.3f}\n")
        
        f.write("\nCAPTURABLE DEBRIS CHARACTERISTICS\n")
        f.write("-" * 20 + "\n")
        f.write("Average values for capturable debris:\n")
        for col in ['altitude_km', 'mass_kg', 'orbital_velocity', 'orbital_energy']:
            mean_val = capturable[col].mean()
            std_val = capturable[col].std()
            f.write(f"- {col}: {mean_val:.2f} ± {std_val:.2f}\n")
    
    print("\nEnhanced analysis complete! Results saved in 'results' folder:")
    print("- results/feature_importance.png (Orbital parameter importance)")
    print("- results/debris_distribution.png (Advanced distribution analysis)")
    print("- results/debris_analysis.txt (Comprehensive findings)")

if __name__ == "__main__":
    analyze_debris()
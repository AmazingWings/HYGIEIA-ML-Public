"""
Enhanced Model Validation for Space Debris Capture
Author: Shriyans Dwivedi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def generate_validation_metrics():
    # Load and prepare data
    print("Loading data...")
    data = pd.read_csv('space_debris_data.csv')
    
    # Prepare features and target
    features = ['altitude_km', 'mass_kg', 'orbital_period_hours', 'inclination_deg']
    X = data[features]
    
    # Create target variable based on ESA guidelines
    data['capturable'] = (
        (data['altitude_km'] <= 1000) &  # Low Earth Orbit focus
        (data['mass_kg'] <= 5000) &      # Mass limit for feasible capture
        (data['size_category'] != 'small')  # Exclude small debris
    )
    y = data['capturable']
    
    # Initialize model with ESA-compliant parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    # 1. K-fold Cross Validation
    print("Performing cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='balanced_accuracy')
    
    # 2. Statistical Significance Testing
    baseline_accuracy = 0.65  # Traditional method accuracy
    t_stat, p_value = stats.ttest_1samp(cv_scores, baseline_accuracy)
    
    # 3. Performance Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(cv_scores * 100, bins=10, edgecolor='black')
    plt.axvline(cv_scores.mean() * 100, color='red', linestyle='dashed', linewidth=2)
    plt.title('Model Performance Distribution Across Cross-Validation')
    plt.xlabel('Balanced Accuracy Score (%)')
    plt.ylabel('Frequency')
    plt.savefig('cross_validation_distribution.png')
    plt.close()
    
    # Generate Report
    report = {
        'Cross Validation Mean': cv_scores.mean() * 100,
        'Cross Validation Std': cv_scores.std() * 100,
        'P-value vs Baseline': p_value,
        'Statistical Significance': p_value < 0.05,
        'Individual Fold Scores': (cv_scores * 100).tolist()
    }
    
    # Save results
    with open('model_validation_report.txt', 'w') as f:
        f.write("Space Debris Capture Model Validation Report\n")
        f.write("=========================================\n\n")
        f.write("1. Cross-Validation Results\n")
        f.write("---------------------------\n")
        f.write(f"Mean Balanced Accuracy: {report['Cross Validation Mean']:.2f}%\n")
        f.write(f"Standard Deviation: ±{report['Cross Validation Std']:.2f}%\n\n")
        
        f.write("2. Statistical Significance Analysis\n")
        f.write("----------------------------------\n")
        f.write(f"Baseline Traditional Method: {baseline_accuracy*100:.2f}%\n")
        f.write(f"P-value: {report['P-value vs Baseline']:.4e}\n")
        f.write(f"Statistically Significant: {report['Statistical Significance']}\n\n")
        
        f.write("3. Individual Fold Performance\n")
        f.write("----------------------------\n")
        for i, score in enumerate(report['Individual Fold Scores'], 1):
            f.write(f"Fold {i}: {score:.2f}%\n")
    
    # Feature Importance Analysis
    print("Analyzing feature importance...")
    model.fit(X, y)
    importance_scores = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)
    
    # Save feature importance
    with open('model_validation_report.txt', 'a') as f:
        f.write("\n4. Feature Importance Analysis\n")
        f.write("---------------------------\n")
        for _, row in feature_importance.iterrows():
            f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Feature Importance for Space Debris Capture')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nValidation analysis complete. Results saved to 'model_validation_report.txt'")
    print(f"Overall Model Performance: {report['Cross Validation Mean']:.2f}% ± {report['Cross Validation Std']:.2f}%")

if __name__ == "__main__":
    generate_validation_metrics()
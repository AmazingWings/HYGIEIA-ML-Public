"""
Hygieia ML Model - Space Debris Capture Analysis
Author: Shriyans Dwivedi
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DebrisClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.size_categories = {'small': 0.05, 'medium': 0.5, 'large': 2.0}
        
    def prepare_data(self, data_path):
        """Load and prepare SARDINE debris data"""
        print("Loading debris data...")
        df = pd.read_csv(data_path)
        
        # Convert categories to numerical values
        df['size_m'] = df['size_category'].map(self.size_categories)
        
        # Calculate orbital velocity
        df['velocity_ms'] = (2 * np.pi * (df['altitude_km'] * 1000)) / (df['orbital_period_hours'] * 3600)
        
        # Determine capturability
        df['capturable'] = ((df['size_m'] <= 0.15) & (df['velocity_ms'] <= 0.5)).astype(int)
        
        # Select features
        features = ['altitude_km', 'mass_kg', 'orbital_period_hours', 'inclination_deg']
        X = df[features]
        y = df['capturable']
        
        return X, y, df
        
        # Scale features
        X = df[self.features]
        y = df['capturable']
        
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

    def train_and_evaluate(self, X, y):
        """Train and evaluate the model"""
        print("\nTraining ML model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"\nModel Accuracy: {accuracy:.1f}%")
        print("\nDetailed Performance:")
        print(classification_report(y_test, y_pred))
        
        return y_test, y_pred

    def create_visualizations(self, X, y, y_pred, df):
        """Generate analysis visualizations"""
        os.makedirs('results', exist_ok=True)
        
        # 1. Feature Importance
        plt.figure(figsize=(10, 6))
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title('Most Important Factors for Debris Capture')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        plt.close()
        
        # 2. Capture Distribution
        plt.figure(figsize=(10, 6))
        capturable = df[df['capturable'] == 1]
        plt.scatter(capturable['altitude_km'], capturable['mass_kg'], 
                   alpha=0.5, label='Capturable')
        plt.xlabel('Altitude (km)')
        plt.ylabel('Mass (kg)')
        plt.title('Distribution of Capturable Debris')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/capture_distribution.png')
        plt.close()
        
        # Save summary
        with open('results/ml_analysis.txt', 'w') as f:
            f.write("HYGIEIA DEBRIS CAPTURE ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total debris analyzed: {len(df)}\n")
            f.write(f"Capturable debris: {df['capturable'].sum()} ({df['capturable'].mean()*100:.1f}%)\n\n")
            f.write("Most important factors:\n")
            for _, row in importance.iterrows():
                f.write(f"- {row['feature']}: {row['importance']:.3f}\n")

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("HYGIEIA ADVANCED ML MODEL TRAINING")
    print("Integrating SARDINE V1 Orbital Database")
    print("="*70)
    
    # Initialize model
    model = HygeiaMLModel()
    
    # Load SARDINE data
    data_path = 'space_debris_data.csv'
    df = model.load_sardine_data(data_path)
    
    if df is None:
        print("Error: Could not load SARDINE data. Using synthetic data instead...")
        # Generate synthetic data
        n_samples = 1000
        np.random.seed(42)
        
        synthetic_data = pd.DataFrame({
            'size': np.random.lognormal(mean=-2.5, sigma=0.8, size=n_samples),
            'velocity': np.random.exponential(scale=0.3, size=n_samples),
            'distance': np.random.uniform(5, 100, size=n_samples),
            'tumble_rate': np.random.uniform(0, 60, size=n_samples),
            'altitude': np.random.normal(500, 100, size=n_samples),
            'inclination': np.random.uniform(0, 180, size=n_samples),
            'albedo': np.random.uniform(0.1, 0.9, size=n_samples),
            'mass': np.random.lognormal(mean=2, sigma=1, size=n_samples)
        })
        
        # Clip values to realistic ranges
        synthetic_data['size'] = synthetic_data['size'].clip(0.01, 0.5)
        synthetic_data['velocity'] = synthetic_data['velocity'].clip(0.01, 2.0)
        synthetic_data['mass'] = synthetic_data['mass'].clip(0.1, 100)
        
        df = synthetic_data
    
    # Preprocess data
    X, y = model.preprocess_data(df)
    
    # Train and evaluate model
    X_test, y_test, y_pred = model.train_model(X, y)
    
    # Analyze feature importance
    importances = model.analyze_feature_importance()
    
    # Create visualizations
    model.visualize_confusion_matrix(y_test, y_pred)
    
    # Save results
    model.save_model_summary(importances)
    
    print("\n" + "="*70)
    print("✅ ML MODEL TRAINING COMPLETE")
    print("="*70)
    print("\nResults saved:")
    print("  📄 results/ml_model_summary.txt")
    print("  📊 results/feature_importance.png")
    print("  📊 results/confusion_matrix.png")

if __name__ == "__main__":
    main()
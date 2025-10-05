# Hygieia AI Space Debris Detection System
**Author:** Shriyans Dwivedi, Green Hope High School

## 🛰️ Project Overview
This project combines two advanced space debris analysis systems:
1. **Hygieia Simulation**: Real-time debris detection and capture simulation
2. **SARDINE V1 Integration**: Real orbital debris database for training

## 🚀 Getting Started Guide

### Prerequisites
- Python 3.x installed
- Git (for cloning the repository)
- 8GB RAM minimum recommended
- Terminal/Command Prompt access

### Step-by-Step Setup

1. **Clone or Download the Repository**
   ```bash
   git clone [your-repository-url]
   cd hygieia-simulation
   ```

2. **Set Up Python Environment**
   ```bash
   # Create a new virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # Install all required packages
   pip install -r requirements.txt
   ```

### Running the Analysis

1. **Run the Enhanced Debris Analysis**
   ```bash
   python debris_analysis.py
   ```
   This will:
   - Load and process the debris data
   - Train the ML model
   - Generate visualizations
   - Save results in the 'results' folder

2. **Run the Simulation**
   ```bash
   python hygieia_simulation.py
   ```
   This will:
   - Start the debris capture simulation
   - Run real-time detection
   - Generate performance metrics

### Output Files
After running the analysis, check the `results/` folder for:
- `debris_analysis.txt`: Detailed analysis report
- `feature_importance.png`: ML model insights
- `debris_distribution.png`: Visual data analysis

### Troubleshooting
- If you see "Module not found" errors:
  ```bash
  pip install -r requirements.txt --upgrade
  ```
- If you get memory errors:
  - Close other applications
  - Ensure you have at least 8GB RAM available

### Development Commands
```bash
# Update dependencies
pip freeze > requirements.txt

# Run tests
python -m pytest tests/

# Format code
black .

# Check code style
flake8
```

## 🧠 Enhanced ML Analysis

### Advanced Orbital Mechanics Integration (`debris_analysis.py`)
- Sophisticated orbital parameter calculations:
  - Orbital velocity and escape velocity at altitude
  - Orbital energy computation
  - Angular velocity analysis
  - Relative velocity potential assessment
- Multi-parameter capture criteria:
  - Size constraints (≤ 15cm)
  - Operational altitude range (200-2000 km)
  - Relative velocity thresholds
  - Orbital energy efficiency zones

## 🤖 Machine Learning System: Technical Deep Dive

### 1. Problem Definition
- **Objective**: Identify space debris objects that can be safely captured by the Hygieia system
- **Type**: Binary Classification (capturable vs. non-capturable)
- **Success Metrics**:
  - High precision (minimize false positives for safety)
  - Strong recall (identify all potentially capturable objects)
  - Reliable confidence scores for decision-making

### 2. Data Processing Pipeline
#### 2.1 Data Collection
- Source: SARDINE V1 debris database
- Features: Orbital parameters, physical characteristics
- Sample size: 34,999 debris objects
- Class distribution: Imbalanced (0.9% capturable)

#### 2.2 Data Preprocessing
1. **Missing Value Handling**:
   - Null detection and removal
   - Statistical imputation where appropriate
   - Validation of data completeness

2. **Feature Engineering**:
   - **Basic Parameters**:
     * Altitude (km)
     * Mass (kg)
     * Orbital period (hours)
     * Inclination (degrees)
   
   - **Physics-Based Features**:
     * Orbital velocity calculation
     * Escape velocity at altitude
     * Gravitational potential energy
     * Kinetic energy analysis
   
   - **Advanced Metrics**:
     * Angular velocity computation
     * Relative velocity potential
     * Energy-mass ratios
     * Orbital stability indicators

3. **Data Transformation**:
   - StandardScaler normalization
   - Categorical encoding
   - Feature correlation analysis
   - Outlier detection and handling

### 3. Model Architecture
#### 3.1 Algorithm Selection
- **Chosen Model**: Random Forest Classifier
  - Ensemble of 500 decision trees
  - Robust to outliers and noise
  - Handles non-linear relationships
  - Provides feature importance rankings

#### 3.2 Model Configuration
- **Core Parameters**:
  ```python
  RandomForestClassifier(
      n_estimators=500,
      max_depth=15,
      min_samples_split=5,
      min_samples_leaf=2,
      max_features='sqrt',
      class_weight='balanced'
  )
  ```

#### 3.3 Training Strategy
1. **Data Split**:
   - 80% Training data
   - 20% Testing data
   - Stratified sampling for class balance

2. **Cross-validation**:
   - 5-fold stratified CV
   - Performance metric: F1-score
   - Confidence threshold validation

### 4. Optimization Process
#### 4.1 Hyperparameter Tuning
- **Grid Search Parameters**:
  - n_estimators: [400, 500, 600]
  - max_depth: [10, 15, 20]
  - min_samples_split: [4, 5, 6]
  - min_samples_leaf: [1, 2, 4]

#### 4.2 Model Selection Criteria
- Cross-validation scores
- F1-score optimization
- ROC-AUC evaluation
- Confidence threshold analysis

### 5. Performance Metrics
#### 5.1 Classification Metrics
- Accuracy: 100.0%
- Precision: 100.0%
- Recall: 98.7%
- F1-Score: 99.3%
- ROC-AUC: 100.0%

#### 5.2 Reliability Metrics
- High confidence predictions: 100%
- Cross-validation stability: ±1.7%
- Feature importance reliability
- Prediction confidence thresholds

### 6. Model Deployment
#### 6.1 Integration Process
- Model serialization for deployment
- API endpoint creation
- Real-time prediction pipeline
- Performance monitoring setup

#### 6.2 Production Environment
- Environment configuration
- Dependency management
- Error handling and logging
- Scalability considerations

### 7. Monitoring and Maintenance
#### 7.1 Performance Monitoring
- Real-time accuracy tracking
- Prediction confidence monitoring
- System resource utilization
- Response time analysis

#### 7.2 Model Updates
- Regular retraining schedule
- Data drift detection
- Performance degradation alerts
- Version control and rollback capability

#### 7.3 Quality Assurance
- Automated testing pipeline
- Validation data updates
- Performance benchmark tracking
- Security and compliance checks

### 8. Future Improvements
#### 8.1 Model Enhancements
- Deep learning integration potential
- Additional feature engineering
- Real-time orbital data integration
- Enhanced confidence scoring

#### 8.2 System Optimization
- Parallel processing implementation
- GPU acceleration potential
- Memory optimization
- Response time improvements

### 9. Usage Examples
```python
# Example: Predicting debris capturability
from debris_analysis import analyze_debris

# Load and analyze debris data
results = analyze_debris('space_debris_data.csv')

# Access prediction results
capturable_objects = results['capturable_debris']
confidence_scores = results['confidence_scores']
feature_importance = results['feature_importance']
```

### 10. Validation Results
#### 10.1 Model Reliability
- Perfect classification accuracy (100%)
- High confidence predictions
- Robust cross-validation performance
- Strong feature importance stability

#### 10.2 Operational Validation
- Tested with real orbital scenarios
- Validated against historical data
- Performance in edge cases
- Integration test results

### 11. Technical Requirements
- Python 3.x
- Scientific Computing Stack:
  * NumPy: Numerical computations
  * Pandas: Data manipulation
  * Scikit-learn: ML algorithms
  * Matplotlib/Seaborn: Visualization
- Hardware Requirements:
  * Minimum 8GB RAM
  * Multi-core processor recommended
  * GPU optional for future enhancements

#### Model Optimization
1. **Hyperparameter Tuning**:
   - Grid search optimization
   - Cross-validated performance metrics
   - Parameters optimized:
     * Number of estimators (400-600)
     * Tree depth (10-20 levels)
     * Min samples for splits (4-6)
     * Min samples per leaf (1-4)

2. **Validation Strategy**:
   - Stratified 5-fold cross-validation
   - Balanced class weights
   - F1-score optimization
   - High confidence thresholding

3. **Performance Metrics**:
   - Accuracy: Model's overall correctness
   - Precision: Reliability of positive predictions
   - Recall: Ability to find all capturable debris
   - F1-Score: Harmonic mean of precision/recall
   - ROC-AUC: Classification quality
   - Confidence Scores: Prediction reliability

#### Data Processing Pipeline
1. **Preprocessing**:
   - StandardScaler normalization
   - Missing value handling
   - Outlier detection
   - Feature correlation analysis

2. **Training Process**:
   - 80-20 train-test split
   - Stratified sampling
   - Cross-validated training
   - Best model selection

3. **Prediction System**:
   - Probability-based classification
   - Confidence thresholding
   - Multi-parameter validation
   - Uncertainty quantification

### Comprehensive Visualization Suite
- **Capture Analysis Dashboard**:
  - Total vs. Capturable debris comparison
  - Size distribution analysis
  - Mission success probability
- **Orbital Parameter Visualization**:
  - Altitude-Velocity distribution plots
  - Energy-Mass relationship analysis
  - Feature importance rankings
- **Statistical Analysis**:
  - Cross-validation performance metrics
  - Classification accuracy reports
  - Capture criteria statistics

## 📊 Enhanced Output Files

The `results/` folder now contains:
- `debris_analysis.txt`: Comprehensive analysis including:
  - Total debris analyzed
  - Capturable object statistics
  - Model performance metrics
  - Cross-validation results
  - Key orbital parameters

- `debris_distribution.png`: Multi-panel visualization:
  - Debris capture potential
  - Size distribution
  - Altitude vs. Velocity plot
  - Energy vs. Mass analysis

- `feature_importance.png`: Enhanced visualization of:
  - Orbital parameter influence
  - Capture criteria impact
  - Feature rankings

## 🔧 Technical Specifications

### Hygieia Vehicle
- 3U CubeSat (10cm × 10cm × 30cm)
- Mass: 15 kg
- Delta-V: 2.63 m/s
- AI System: Raspberry Pi 5
- Camera: Sony IMX500

### Enhanced Capture Constraints
- Size Constraints:
  - Maximum debris size: 15 cm
  - Mass consideration: Based on orbital energy

- Orbital Parameters:
  - Operational altitude: 200-2000 km
  - Relative velocity: ≤ 0.5 m/s from mean
  - Energy threshold: Within 10% of mean orbital energy

- Detection Capabilities:
  - Range: 10-50 meters
  - Angular velocity tracking
  - Energy profile analysis

### Model Performance
- Classification accuracy: ~100%
- Cross-validation stability: High
- Feature importance ranking
- Capture prediction reliability

## 📝 Research Integration
This enhanced simulation system provides sophisticated empirical validation for the research paper "Designing a Sustainable Space Environment: A Vehicle-Based Approach to Space Debris Management". The integration of advanced orbital mechanics and machine learning techniques offers deeper insights into debris capture strategies and mission planning.

## 🔄 Continuous Improvements
- Enhanced orbital parameter analysis
- Additional feature engineering
- Real-time prediction capabilities
- Integration with live tracking data
- Optimization of capture strategies
- Extended visualization capabilities
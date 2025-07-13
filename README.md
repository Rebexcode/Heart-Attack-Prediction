# Heart Attack Prediction App for Hypertensive Patients

A comprehensive Streamlit application that provides heart attack risk assessment specifically designed for patients with hypertension (high blood pressure ≥ 140 mmHg).

## Features

### 🔍 Screen 1: Data Analysis Dashboard

- Interactive data exploration and visualization
- Real-time filtering capabilities (age, gender, chest pain type)
- Key statistics and metrics display
- Correlation matrix analysis
- Distribution plots for key health indicators

### 🎯 Screen 2: Prediction Interface

- User-friendly form for patient data input
- Real-time risk assessment using machine learning models
- Detailed risk factor analysis
- Personalized recommendations based on risk level
- Comprehensive validation for hypertensive patients only

### 🤖 Screen 3: Chatbot Recommendation System

- Interactive health recommendations chatbot
- Personalized advice based on prediction results
- Topics covered:
  - Diet and nutrition guidance
  - Exercise recommendations
  - Medication management
  - Stress management techniques
  - Blood pressure monitoring
  - Emergency symptom recognition

## Installation

1. **Clone or download** this repository to your local machine.

2. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the dataset**: Make sure `balanced_heart_disease_dataset.csv` is in the same directory as `app.py`.

## Running the Application

1. **Navigate to the project directory**:

   ```bash
   cd heart-attack-predicton
   ```

2. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and go to the URL displayed in the terminal (usually `http://localhost:8501`).

## Usage Guide

### Data Analysis Dashboard

- Use the sidebar filters to explore different patient subgroups
- View key statistics and distribution patterns
- Analyze correlations between health indicators
- Export charts for presentations or reports

### Risk Prediction

- Fill in all required patient information
- Ensure the patient has hypertension (BP ≥ 140 mmHg)
- Click "Predict Heart Attack Risk" to get results
- Review the detailed risk factor analysis
- Follow the provided recommendations

### Health Chatbot

- Ask questions about heart health management
- Get personalized recommendations based on your prediction
- Learn about diet, exercise, medications, and stress management
- Understand emergency warning signs

## Input Features

The prediction model requires the following patient information:

- **Age**: Patient's age in years (18-100)
- **Gender**: Male or Female
- **Chest Pain Type**:
  - Typical Angina
  - Atypical Angina
  - Non-Anginal Pain
  - Asymptomatic
- **Resting Blood Pressure**: ≥140 mmHg (hypertensive range)
- **Cholesterol Level**: Total cholesterol in mg/dL
- **Fasting Blood Sugar**: >120 mg/dL (Yes/No)
- **Resting ECG Results**:
  - Normal
  - ST-T Wave Abnormality
  - Left Ventricular Hypertrophy
- **Maximum Heart Rate**: Achieved during exercise
- **Exercise Induced Angina**: Yes/No
- **ST Depression**: Exercise-induced ST depression
- **ST Segment Slope**:
  - Upsloping
  - Flat
  - Downsloping

## Model Information

The application uses a simplified prediction model based on the machine learning approach from the accompanying Jupyter notebook. In a production environment, this would be replaced with the actual trained models (XGBoost, Random Forest, SVM) saved as pickle files.

### Model Features

- **Target Population**: Hypertensive patients only
- **Prediction Type**: Binary classification (High Risk / Low Risk)
- **Output**: Risk probability, risk score, and detailed recommendations

## Technical Details

### Built With

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Seaborn/Matplotlib**: Statistical visualizations
- **Scikit-learn**: Machine learning utilities
- **XGBoost**: Gradient boosting framework

### Architecture

- **Multi-screen application** with sidebar navigation
- **Session state management** for chat history and predictions
- **Responsive design** with mobile-friendly interface
- **Real-time data filtering** and visualization updates

## Important Notes

### Medical Disclaimer

⚠️ **This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.**

### Data Requirements

- The application requires the dataset file `balanced_heart_disease_dataset.csv`
- Patient data must indicate hypertension (BP ≥ 140 mmHg)
- All input fields are required for accurate predictions

### Performance Considerations

- The application caches data loading for better performance
- Large datasets may take time to load initially
- Visualizations are optimized for web display

## Troubleshooting

### Common Issues

1. **Dataset not found**:

   - Ensure `balanced_heart_disease_dataset.csv` is in the same directory as `app.py`
   - Check file permissions and path

2. **Package installation errors**:

   - Use a virtual environment
   - Update pip: `pip install --upgrade pip`
   - Install packages individually if batch installation fails

3. **Streamlit not starting**:

   - Check if port 8501 is available
   - Try a different port: `streamlit run app.py --server.port 8502`

4. **Prediction errors**:
   - Ensure all form fields are filled
   - Verify blood pressure is ≥140 mmHg
   - Check input ranges for all numeric fields

## Future Enhancements

- Integration with real trained ML models
- Export functionality for predictions and reports
- Multi-language support
- Mobile app version
- Electronic health record integration
- Batch prediction capabilities

## Support

For technical issues or questions about the application, please refer to the Jupyter notebook documentation or consult the development team.

---

**Version**: 1.0  
**Last Updated**: January 2025  
**Compatible with**: Python 3.8+, Streamlit 1.28+

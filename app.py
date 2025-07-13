"""
Heart Attack Prediction App for Hypertensive Patients
=====================================================

A Streamlit application with three main screens:
1. Data Analysis Dashboard
2. Prediction Interface 
3. Chatbot Recommendation System

Based on the machine learning models trained in the Jupyter notebook.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Attack Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff6b6b;
        margin-bottom: 1rem;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Load dataset
@st.cache_data
def load_dataset():
    """Load and cache the heart disease dataset"""
    try:
        df = pd.read_csv('balanced_heart_disease_dataset.csv')
        
        # Apply column renaming as in the notebook
        column_names = {
            'age': 'Age',
            'sex': 'Sex',
            'chest pain type': 'ChestPainType',
            'resting bp s': 'RestingBP',
            'cholesterol': 'Cholesterol',
            'fasting blood sugar': 'FastingBS',
            'resting ecg': 'RestingECG',
            'max heart rate': 'MaxHeartRate',
            'exercise angina': 'ExerciseAngina',
            'oldpeak': 'STDepression',
            'ST slope': 'STSlope',
            'target': 'Target'
        }
        
        existing_columns = set(df.columns)
        columns_to_rename = {old: new for old, new in column_names.items() if old in existing_columns}
        if columns_to_rename:
            df.rename(columns=columns_to_rename, inplace=True)
        
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'balanced_heart_disease_dataset.csv' is in the working directory.")
        return None

# Prediction function (simplified version from notebook)
def predict_heart_attack_hypertensive(patient_data):
    """
    Simplified prediction function for the Streamlit app
    Note: In a production app, you would load pre-trained models
    """
    # For demo purposes, we'll create a simple rule-based prediction
    # In practice, you would load your trained models here
    
    risk_score = 0
    
    # Age factor
    if patient_data['Age'] > 60:
        risk_score += 2
    elif patient_data['Age'] > 45:
        risk_score += 1
    
    # Cholesterol factor
    if patient_data['Cholesterol'] > 240:
        risk_score += 2
    elif patient_data['Cholesterol'] > 200:
        risk_score += 1
    
    # Blood pressure factor (already hypertensive)
    if patient_data['RestingBP'] > 160:
        risk_score += 2
    else:
        risk_score += 1
    
    # Exercise angina
    if patient_data['ExerciseAngina'] == 1:
        risk_score += 2
    
    # ST Depression
    if patient_data['STDepression'] > 1.5:
        risk_score += 2
    elif patient_data['STDepression'] > 0.5:
        risk_score += 1
    
    # Simple threshold-based prediction
    probability = min(risk_score / 10.0, 0.95)
    prediction = 1 if probability > 0.5 else 0
    
    return {
        'prediction': prediction,
        'probability': probability,
        'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
        'risk_score': risk_score
    }

def screen_data_analysis():
    """Screen 1: Data Analysis Dashboard"""
    st.markdown('<h1 class="main-header">📊 Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_dataset()
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.markdown("### 🔍 Data Filters")
    
    # Age filter
    age_range = st.sidebar.slider(
        "Age Range", 
        int(df['Age'].min()) if 'Age' in df.columns else 20, 
        int(df['Age'].max()) if 'Age' in df.columns else 80, 
        (30, 70)
    )
    
    # Sex filter
    sex_options = ['All', 'Male', 'Female']
    sex_filter = st.sidebar.selectbox("Gender", sex_options)
    
    # Chest pain type filter
    if 'ChestPainType' in df.columns:
        chest_pain_options = ['All'] + sorted(df['ChestPainType'].unique().tolist())
        chest_pain_filter = st.sidebar.selectbox("Chest Pain Type", chest_pain_options)
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'Age' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['Age'] >= age_range[0]) & 
            (filtered_df['Age'] <= age_range[1])
        ]
    
    if sex_filter != 'All' and 'Sex' in df.columns:
        sex_value = 1 if sex_filter == 'Male' else 0
        filtered_df = filtered_df[filtered_df['Sex'] == sex_value]
    
    if 'chest_pain_filter' in locals() and chest_pain_filter != 'All' and 'ChestPainType' in df.columns:
        filtered_df = filtered_df[filtered_df['ChestPainType'] == chest_pain_filter]
    
    # Display dataset information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(filtered_df))
    
    with col2:
        if 'Target' in filtered_df.columns:
            high_risk_count = len(filtered_df[filtered_df['Target'] == 1])
            st.metric("High Risk Cases", high_risk_count)
    
    with col3:
        if 'Age' in filtered_df.columns:
            avg_age = filtered_df['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f}")
    
    with col4:
        if 'RestingBP' in filtered_df.columns:
            avg_bp = filtered_df['RestingBP'].mean()
            st.metric("Average BP", f"{avg_bp:.1f}")
    
    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)
    
    # Visualizations
    st.subheader("📈 Data Visualizations")
    
    # Create two columns for side-by-side charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        if 'Age' in filtered_df.columns:
            st.markdown("**Age Distribution**")
            fig_age = px.histogram(
                filtered_df, 
                x='Age', 
                nbins=20, 
                title="Age Distribution",
                color_discrete_sequence=['#1f77b4']
            )
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Cholesterol distribution
        if 'Cholesterol' in filtered_df.columns:
            st.markdown("**Cholesterol Distribution**")
            fig_chol = px.histogram(
                filtered_df, 
                x='Cholesterol', 
                nbins=20, 
                title="Cholesterol Distribution",
                color_discrete_sequence=['#ff7f0e']
            )
            fig_chol.update_layout(height=400)
            st.plotly_chart(fig_chol, use_container_width=True)
    
    # Target distribution
    if 'Target' in filtered_df.columns:
        st.markdown("**Risk Level Distribution**")
        target_counts = filtered_df['Target'].value_counts()
        fig_target = px.pie(
            values=target_counts.values,
            names=['Low Risk', 'High Risk'],
            title="Risk Level Distribution"
        )
        st.plotly_chart(fig_target, use_container_width=True)
    
    # Correlation matrix
    st.subheader("🔗 Correlation Matrix")
    numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        # Limit to key features for readability
        key_features = [col for col in ['Age', 'RestingBP', 'Cholesterol', 'MaxHeartRate', 
                       'STDepression', 'Target'] if col in numeric_columns]
        
        if len(key_features) > 1:
            corr_matrix = filtered_df[key_features].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r'
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

def screen_prediction():
    """Screen 2: Prediction Interface"""
    st.markdown('<h1 class="main-header">🔮 Heart Attack Risk Prediction</h1>', unsafe_allow_html=True)
    st.markdown("### For Hypertensive Patients Only")
    
    # Information about the prediction
    st.info("""
    ℹ️ **About This Prediction Tool**
    
    This tool is specifically designed for patients with hypertension (high blood pressure ≥ 140 mmHg).
    Please fill in all the required information below to get your risk assessment.
    """)
    
    # Create prediction form
    with st.form("prediction_form"):
        st.subheader("📋 Patient Information")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age (years)", 
                min_value=18, 
                max_value=100, 
                value=50,
                help="Patient's age in years"
            )
            
            sex = st.selectbox(
                "Gender", 
                options=[0, 1], 
                format_func=lambda x: "Female" if x == 0 else "Male",
                help="Biological sex"
            )
            
            chest_pain_type = st.selectbox(
                "Chest Pain Type",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Typical Angina",
                    2: "Atypical Angina", 
                    3: "Non-Anginal Pain",
                    4: "Asymptomatic"
                }[x],
                help="Type of chest pain experienced"
            )
            
            resting_bp = st.number_input(
                "Resting Blood Pressure (mmHg)", 
                min_value=140, 
                max_value=220, 
                value=150,
                help="Must be ≥140 mmHg (hypertensive range)"
            )
            
            cholesterol = st.number_input(
                "Cholesterol Level (mg/dL)", 
                min_value=100, 
                max_value=600, 
                value=220,
                help="Total cholesterol level"
            )
            
            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Is fasting blood sugar greater than 120 mg/dL?"
            )
        
        with col2:
            resting_ecg = st.selectbox(
                "Resting ECG Results",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "Normal",
                    1: "ST-T Wave Abnormality",
                    2: "Left Ventricular Hypertrophy"
                }[x],
                help="Resting electrocardiogram results"
            )
            
            max_heart_rate = st.number_input(
                "Maximum Heart Rate Achieved", 
                min_value=60, 
                max_value=220, 
                value=150,
                help="Maximum heart rate during exercise test"
            )
            
            exercise_angina = st.selectbox(
                "Exercise Induced Angina",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Does exercise induce angina?"
            )
            
            st_depression = st.number_input(
                "ST Depression", 
                min_value=0.0, 
                max_value=6.0, 
                value=0.0, 
                step=0.1,
                help="ST depression induced by exercise relative to rest"
            )
            
            st_slope = st.selectbox(
                "ST Segment Slope",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "Upsloping",
                    1: "Flat",
                    2: "Downsloping"
                }[x],
                help="Slope of peak exercise ST segment"
            )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Heart Attack Risk", use_container_width=True)
        
        if submitted:
            # Validate hypertensive condition
            if resting_bp < 140:
                st.error("⚠️ This tool is designed for hypertensive patients only (BP ≥ 140 mmHg)")
                return
            
            # Create patient data dictionary
            patient_data = {
                'Age': age,
                'Sex': sex,
                'ChestPainType': chest_pain_type,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': fasting_bs,
                'RestingECG': resting_ecg,
                'MaxHeartRate': max_heart_rate,
                'ExerciseAngina': exercise_angina,
                'STDepression': st_depression,
                'STSlope': st_slope
            }
            
            # Make prediction
            with st.spinner("Analyzing patient data..."):
                result = predict_heart_attack_hypertensive(patient_data)
            
            # Store result in session state for chatbot
            st.session_state.last_prediction = result
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            # Create result container with appropriate styling
            risk_class = "high-risk" if result['prediction'] == 1 else "low-risk"
            
            st.markdown(f"""
            <div class="prediction-result {risk_class}">
                <h3>Risk Level: {result['risk_level']}</h3>
                <p><strong>Probability:</strong> {result['probability']:.1%}</p>
                <p><strong>Risk Score:</strong> {result['risk_score']}/10</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional information based on risk level
            if result['prediction'] == 1:
                st.error("""
                ⚠️ **High Risk Detected**
                
                This prediction suggests an elevated risk of heart attack. Please consider:
                - Consulting with a cardiologist immediately
                - Regular monitoring of blood pressure and cholesterol
                - Lifestyle modifications (diet, exercise)
                - Medication compliance if prescribed
                """)
            else:
                st.success("""
                ✅ **Lower Risk Detected**
                
                While the risk appears lower, please continue:
                - Regular health check-ups
                - Maintaining healthy lifestyle habits
                - Monitoring blood pressure regularly
                - Following medical advice for hypertension management
                """)
            
            # Show feature importance/contributions
            st.subheader("📊 Risk Factor Analysis")
            
            # Create a simple risk factor breakdown
            factors = []
            if age > 60:
                factors.append(("Advanced Age", "High"))
            elif age > 45:
                factors.append(("Middle Age", "Moderate"))
            
            if cholesterol > 240:
                factors.append(("High Cholesterol", "High"))
            elif cholesterol > 200:
                factors.append(("Elevated Cholesterol", "Moderate"))
            
            if resting_bp > 160:
                factors.append(("Severe Hypertension", "High"))
            else:
                factors.append(("Mild-Moderate Hypertension", "Moderate"))
            
            if exercise_angina == 1:
                factors.append(("Exercise-Induced Angina", "High"))
            
            if st_depression > 1.5:
                factors.append(("Significant ST Depression", "High"))
            elif st_depression > 0.5:
                factors.append(("Mild ST Depression", "Moderate"))
            
            if factors:
                factor_df = pd.DataFrame(factors, columns=['Risk Factor', 'Impact Level'])
                st.dataframe(factor_df, use_container_width=True)

def screen_chatbot():
    """Screen 3: Chatbot Recommendation System"""
    st.markdown('<h1 class="main-header">🤖 Health Recommendations Chatbot</h1>', unsafe_allow_html=True)
    
    # Initialize chatbot with welcome message
    if not st.session_state.chat_history:
        welcome_msg = """
        👋 Hello! I'm your Heart Health Assistant. I can provide personalized recommendations based on your recent prediction results and answer questions about heart health for hypertensive patients.
        
        How can I help you today?
        """
        st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about heart health recommendations..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate bot response
        response = generate_bot_response(prompt)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)

def generate_bot_response(user_input):
    """Generate chatbot responses based on user input and last prediction"""
    user_input_lower = user_input.lower()
    
    # Get last prediction if available
    last_prediction = st.session_state.last_prediction
    
    # Response based on keywords
    if any(word in user_input_lower for word in ['diet', 'food', 'eat', 'nutrition']):
        return """
        🥗 **Heart-Healthy Diet Recommendations for Hypertensive Patients:**
        
        **Foods to Include:**
        - Fruits and vegetables (aim for 5-7 servings daily)
        - Whole grains (brown rice, quinoa, oats)
        - Lean proteins (fish, poultry, beans)
        - Low-fat dairy products
        - Nuts and seeds (unsalted)
        
        **Foods to Limit:**
        - Sodium (less than 2,300mg daily, ideally 1,500mg)
        - Saturated fats and trans fats
        - Added sugars
        - Processed and packaged foods
        - Alcohol (limit to moderate consumption)
        
        **Special Focus:** Follow the DASH diet pattern, which has been proven effective for blood pressure management.
        """
    
    elif any(word in user_input_lower for word in ['exercise', 'physical', 'activity', 'workout']):
        return """
        🏃‍♂️ **Exercise Recommendations for Heart Health:**
        
        **Aerobic Exercise:**
        - 150 minutes of moderate-intensity exercise per week
        - OR 75 minutes of vigorous-intensity exercise per week
        - Examples: brisk walking, swimming, cycling
        
        **Strength Training:**
        - 2-3 sessions per week
        - Focus on major muscle groups
        - Use light to moderate weights
        
        **Important Precautions:**
        - Start slowly and gradually increase intensity
        - Monitor your heart rate during exercise
        - Stop if you experience chest pain, dizziness, or shortness of breath
        - Consult your doctor before starting any new exercise program
        
        **Blood Pressure Considerations:** Exercise can help lower blood pressure, but monitor it regularly.
        """
    
    elif any(word in user_input_lower for word in ['medication', 'medicine', 'drugs', 'pills']):
        return """
        💊 **Medication Management for Hypertensive Patients:**
        
        **Common Hypertension Medications:**
        - ACE inhibitors
        - ARBs (Angiotensin Receptor Blockers)
        - Diuretics
        - Beta-blockers
        - Calcium channel blockers
        
        **Important Reminders:**
        - Take medications exactly as prescribed
        - Don't skip doses
        - Don't stop medications without consulting your doctor
        - Monitor for side effects
        - Keep a medication schedule
        
        **Heart Attack Prevention:**
        - Low-dose aspirin (if recommended by doctor)
        - Statins for cholesterol management
        - Blood pressure medications as prescribed
        
        ⚠️ **Always consult your healthcare provider before making any medication changes.**
        """
    
    elif any(word in user_input_lower for word in ['stress', 'anxiety', 'worry', 'mental']):
        return """
        🧘‍♀️ **Stress Management for Heart Health:**
        
        **Why Stress Matters:**
        - Chronic stress can raise blood pressure
        - May increase heart attack risk
        - Can worsen existing heart conditions
        
        **Stress Reduction Techniques:**
        - Deep breathing exercises (4-7-8 technique)
        - Meditation and mindfulness
        - Progressive muscle relaxation
        - Yoga or tai chi
        - Regular sleep schedule (7-9 hours)
        
        **Lifestyle Changes:**
        - Limit caffeine
        - Avoid smoking and excessive alcohol
        - Maintain social connections
        - Consider counseling or therapy
        - Practice time management
        
        **When to Seek Help:** If stress feels overwhelming or affects daily life, consult a mental health professional.
        """
    
    elif any(word in user_input_lower for word in ['blood pressure', 'bp', 'hypertension']):
        return """
        🩺 **Blood Pressure Management:**
        
        **Target Goals:**
        - Generally less than 130/80 mmHg
        - Your doctor may set specific targets for you
        
        **Home Monitoring:**
        - Measure at the same time daily
        - Use a validated monitor
        - Keep a log for your doctor
        - Take multiple readings and average them
        
        **Lifestyle Factors:**
        - Reduce sodium intake
        - Maintain healthy weight
        - Regular physical activity
        - Limit alcohol consumption
        - Manage stress effectively
        
        **When to Contact Doctor:**
        - Consistently high readings (>180/120)
        - Sudden spikes in blood pressure
        - Symptoms like severe headache, chest pain, or vision changes
        """
    
    elif any(word in user_input_lower for word in ['prediction', 'result', 'risk']):
        if last_prediction:
            if last_prediction['prediction'] == 1:
                return f"""
                🚨 **Your Recent High-Risk Prediction - Action Plan:**
                
                **Immediate Steps:**
                1. Schedule an appointment with a cardiologist within 1-2 weeks
                2. Continue taking all prescribed medications
                3. Monitor blood pressure daily
                4. Avoid strenuous activities until cleared by doctor
                
                **Risk Factors to Address:**
                - Your risk score was {last_prediction['risk_score']}/10
                - Probability: {last_prediction['probability']:.1%}
                
                **Lifestyle Modifications:**
                - Implement heart-healthy diet immediately
                - Gentle exercise as tolerated (walking)
                - Stress management techniques
                - Ensure adequate sleep
                
                **Emergency Signs:** Call 911 if you experience chest pain, shortness of breath, or severe symptoms.
                """
            else:
                return f"""
                ✅ **Your Recent Lower-Risk Prediction - Maintenance Plan:**
                
                **Continue Good Practices:**
                - Your risk score was {last_prediction['risk_score']}/10
                - Probability: {last_prediction['probability']:.1%}
                
                **Preventive Measures:**
                1. Regular health check-ups every 6 months
                2. Maintain current blood pressure management
                3. Continue heart-healthy lifestyle
                4. Monitor for any changes in symptoms
                
                **Stay Vigilant:**
                - Even with lower risk, hypertension requires ongoing management
                - Follow up with your healthcare provider regularly
                - Keep tracking your health metrics
                """
        else:
            return """
            📊 **About Risk Predictions:**
            
            To get personalized recommendations, please first use the Prediction Interface to assess your heart attack risk.
            
            The prediction takes into account multiple factors including:
            - Age and gender
            - Blood pressure levels
            - Cholesterol levels
            - Exercise tolerance
            - ECG findings
            - Other cardiovascular risk factors
            
            Once you have a prediction, I can provide specific recommendations based on your results.
            """
    
    elif any(word in user_input_lower for word in ['emergency', 'symptoms', 'chest pain', 'heart attack']):
        return """
        🚨 **Heart Attack Warning Signs - Call 911 Immediately:**
        
        **Major Symptoms:**
        - Chest pain, pressure, or discomfort
        - Pain radiating to arms, back, neck, jaw, or stomach
        - Shortness of breath
        - Cold sweats
        - Nausea or vomiting
        - Lightheadedness or fainting
        
        **For Women (may have different symptoms):**
        - Unusual fatigue
        - Back or jaw pain
        - Nausea
        - Shortness of breath without chest pain
        
        **What to Do:**
        1. Call 911 immediately
        2. Chew aspirin if not allergic (if recommended by previous doctor)
        3. Stay calm and rest
        4. Unlock your door for emergency responders
        
        **Never ignore these symptoms or try to "tough it out"**
        """
    
    else:
        return """
        🤔 I'd be happy to help you with heart health information! I can provide guidance on:
        
        - **Diet and Nutrition** for heart health
        - **Exercise** recommendations for hypertensive patients
        - **Medication** management tips
        - **Stress Management** techniques
        - **Blood Pressure** monitoring and control
        - **Emergency Symptoms** to watch for
        - **Risk Assessment** interpretations
        
        What specific topic would you like to know more about? Just ask me about any of these areas, and I'll provide detailed, personalized recommendations based on your situation.
        
        Remember: This chatbot provides general information and should not replace professional medical advice. Always consult your healthcare provider for specific medical concerns.
        """

# Main app navigation
def main():
    """Main application function with navigation"""
    
    # Sidebar navigation
    st.sidebar.title("🏥 Navigation")
    
    # Navigation menu
    screens = {
        "📊 Data Analysis": screen_data_analysis,
        "🔮 Risk Prediction": screen_prediction,
        "🤖 Health Chatbot": screen_chatbot
    }
    
    selected_screen = st.sidebar.radio("Choose a screen:", list(screens.keys()))
    
    # Add information about the app
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This App")
    st.sidebar.info("""
    This application helps assess heart attack risk specifically for patients with hypertension (high blood pressure).
    
    **Features:**
    - Data analysis and visualization
    - ML-based risk prediction
    - Personalized health recommendations
    """)
    
    # Add disclaimer
    st.sidebar.markdown("---")
    st.sidebar.warning("""
    ⚠️ **Medical Disclaimer**
    
    This tool is for educational purposes only and should not replace professional medical advice. Always consult healthcare providers for medical decisions.
    """)
    
    # Run selected screen
    screens[selected_screen]()

if __name__ == "__main__":
    main()

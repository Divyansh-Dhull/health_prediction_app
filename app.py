import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import numpy as np

# --- 1. ADVANCED UI/UX STYLING WITH CSS & JS ---

# Page Configuration
st.set_page_config(page_title="Health Prediction System 🩺", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for advanced styling, animations, and the contact modal
st.markdown("""
<style>
/* --- General & Fonts --- */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* --- Main Content --- */
.main .block-container {
    padding: 2rem 5rem;
}

/* --- Titles --- */
h1, h2, h3 {
    color: #1a73e8; /* A vibrant blue that works well in both modes */
}

/* --- Buttons --- */
.stButton>button {
    color: #ffffff;
    background-color: #1a73e8;
    border: none;
    padding: 0.8rem 1.6rem;
    border-radius: 0.5rem;
    font-size: 1.1rem;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.stButton>button:hover {
    background-color: #155ab6;
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0,0,0,0.15);
}

/* --- Interactive Disease Cards --- */
.disease-card {
    background-color: var(--streamlit-secondary-background-color);
    color: var(--streamlit-text-color);
    border-radius: 1rem;
    padding: 1.5rem;
    margin: 1rem 0;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease-in-out;
    cursor: pointer;
    border: 1px solid var(--streamlit-faded-text-05); /* Subtle border */
}
.disease-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    border-color: #1a73e8;
}
.disease-card h3 {
    margin-bottom: 0.5rem;
    font-size: 1.5rem;
    color: #1a73e8;
}

/* --- SIDEBAR HOVER FIX --- */
/* This forces the text to be black when hovering over a sidebar item, 
   ensuring visibility against the light hover background. */
.nav-link:hover {
    color: #000000 !important;
}

/* --- Contact Us Icon & Modal --- */
.contact-icon {
    position: fixed;
    top: 25px;
    right: 35px;
    font-size: 2rem;
    cursor: pointer;
    z-index: 1000;
    color: #1a73e8;
    transition: transform 0.2s;
}
.contact-icon:hover {
    transform: scale(1.15);
}

.modal {
    display: none; /* Hidden by default */
    position: fixed;
    z-index: 1001;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.5);
    justify-content: center;
    align-items: center;
}
.modal-content {
    background-color: #ffffff; /* Modal is explicitly white */
    color: #000000; /* Text must be black to be visible on white background */
    padding: 25px 35px;
    border-radius: 1rem;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    width: 90%;
    max-width: 400px;
    text-align: center;
    position: relative;
}
.close-button {
    color: #aaa;
    position: absolute;
    top: 10px;
    right: 20px;
    font-size: 28px;
    font-weight: bold;
    cursor: pointer;
}
.close-button:hover {
    color: #333;
}
</style>

<!-- Contact Us Icon and Modal HTML -->
<div id="contactModal" class="modal">
    <div class="modal-content">
        <span class="close-button" onclick="document.getElementById('contactModal').style.display='none'">&times;</span>
        <h3>Contact Support</h3>
        <hr>
        <p>For any help or support, feel free to reach out to us:</p>
        <p><strong>Email:</strong> support@healthpredict.dummy</p>
        <p><strong>Toll-Free Number:</strong> 1800-000-0000</p>
    </div>
</div>
<div class="contact-icon" onclick="document.getElementById('contactModal').style.display='flex'">
    📧
</div>

<!-- JavaScript to control the modal -->
<script>
var modal = document.getElementById('contactModal');
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}
</script>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        # Fail silently or show a warning, but don't crash the UI demo
        return None, None

diabetes_model, diabetes_scaler = load_model_and_scaler('models/diabetes_model.pkl', 'models/diabetes_scaler.pkl')
heart_model, heart_scaler = load_model_and_scaler('models/heart_model.pkl', 'models/heart_scaler.pkl')
parkinsons_model, parkinsons_scaler = load_model_and_scaler('models/parkinsons_model.pkl', 'models/parkinsons_scaler.pkl')


# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    selected = option_menu(
        'Health Prediction System 🩺',
        ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
        icons=['house-door-fill', 'bi-droplet-fill', 'bi-heart-pulse-fill', 'bi-person-fill'],
        menu_icon="clipboard2-pulse-fill",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "var(--streamlit-secondary-background-color)"},
            "icon": {"color": "#1a73e8", "font-size": "23px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#f0f2f6"},
            # FIX: Added "color": "#000000" to ensure text is visible when selected
            "nav-link-selected": {"background-color": "#e7f3ff", "color": "#000000"},
        }
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "*Disclaimer:* This is an educational tool and not a substitute for professional medical advice. Predictions are for informational purposes only. Always consult a qualified healthcare provider for health concerns."
    )


# --- HOME PAGE with Interactive Cards ---
if selected == 'Home':
    st.title("Welcome to the AI Health Prediction System 🩺")
    st.markdown("Your personal guide to understanding potential health risks. Choose a category below to get started.")

    # Initialize session state for card clicks
    if 'clicked' not in st.session_state:
        st.session_state.clicked = None

    def create_card(disease_name, description):
        if st.button(f"Analyze {disease_name}", key=disease_name, use_container_width=True):
             # When a button is clicked, set it as the 'clicked' state
            st.session_state.clicked = disease_name if st.session_state.clicked != disease_name else None
        st.markdown(f"<div class='disease-card'><h3>{disease_name}</h3><p>{description}</p></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        create_card("Diabetes", "Predict the likelihood of diabetes based on key health metrics.")
    with col2:
        create_card("Heart Disease", "Assess your risk of heart disease using clinical data.")
    with col3:
        create_card("Parkinson's", "Analyze vocal measurements to detect signs of Parkinson's disease.")
    
    st.markdown("---")

    # Display details if a card is "clicked"
    if st.session_state.clicked == "Diabetes":
        with st.expander("About Diabetes", expanded=True):
            st.markdown("""
            *Diabetes Mellitus* is a chronic metabolic disorder characterized by high blood sugar levels. It occurs when the body either doesn't produce enough insulin or can't effectively use the insulin it produces.
            
            ### Most Common & Identifiable Symptoms:
            - *Frequent Urination:* Needing to go to the bathroom more often than usual, especially at night.
            - *Increased Thirst:* Feeling thirsty all the time, even after drinking fluids.
            - *Unexplained Weight Loss:* Losing weight without trying through diet or exercise.
            - *Increased Hunger:* Feeling very hungry even though you are eating.
            - *Blurry Vision:* High blood sugar can affect the lenses in your eyes.
            - *Numb or Tingling Hands and Feet:* A sign of nerve damage caused by high blood sugar.
            - *Fatigue:* Feeling tired and lethargic.
            """)
    
    if st.session_state.clicked == "Heart Disease":
        with st.expander("About Heart Disease", expanded=True):
            st.markdown("""
            *Heart Disease* refers to a range of conditions that affect the heart. These include coronary artery disease, heart failure, arrhythmias, and heart valve problems. It is a leading cause of death worldwide but is often preventable.
            
            ### Most Common & Identifiable Symptoms:
            - *Chest Pain (Angina):* A feeling of pressure, tightness, or squeezing in the chest.
            - *Shortness of Breath:* Difficulty breathing, even at rest or with minimal exertion.
            - *Pain in the Arm or Jaw:* Pain radiating from the chest to the left arm, back, neck, or jaw.
            - *Swelling in Legs, Ankles, and Feet:* Known as edema, this can be a sign of heart failure.
            - *Fatigue:* Unusual or extreme tiredness that doesn't go away with rest.
            - *Dizziness or Lightheadedness:* Feeling faint or about to pass out.
            """)

    if st.session_state.clicked == "Parkinson's":
        with st.expander("About Parkinson's Disease", expanded=True):
            st.markdown("""
            *Parkinson's Disease* is a progressive neurodegenerative disorder that primarily affects dopamine-producing neurons in a specific area of the brain. This leads to a decline in motor function. The predictions here are based on vocal analysis, as Parkinson's can affect speech patterns early on.

            ### Most Common & Identifiable Symptoms:
            - *Tremor:* Shaking that usually begins in a limb, often the hand or fingers, when at rest.
            - *Slowed Movement (Bradykinesia):* Difficulty initiating voluntary movements, making simple tasks difficult and time-consuming.
            - *Rigid Muscles:* Muscle stiffness can occur in any part of the body, limiting the range of motion and causing pain.
            - *Impaired Posture and Balance:* A stooped posture or balance problems.
            - *Speech Changes:* Speaking softly, quickly, slurring, or hesitating before talking. The voice may become more of a monotone.
            - *Changes in Handwriting:* Handwriting may become small and cramped (micrographia).
            """)

# --- PREDICTION PAGES with Detailed Help ---

if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction")
    st.markdown("Enter the values below. Hover over the *(?)* for detailed information on each parameter.")
    
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', help="Enter the total number of times you have been pregnant. *Standard Value:* Typically 0 or more.")
        Glucose = st.text_input('Glucose', help="Your plasma glucose level from a glucose tolerance test. *Normal Range:* < 140 mg/dL. *To Measure:* This requires a blood test ordered by a doctor.")
        SkinThickness = st.text_input('Skin Thickness', help="Triceps skin fold thickness. *Normal Range:* Varies widely, but an average is ~20-30mm. *To Measure:* Requires special calipers; not a home measurement.")
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', help="A score assessing diabetes likelihood based on family history. *Standard Value:* This is a calculated score; an average value might be around 0.4-0.5. Higher values indicate a stronger genetic link.")
    with col2:
        Age = st.text_input('Age', help="Your current age in years. *Standard Value:* N/A.")
        BloodPressure = st.text_input('Blood Pressure', help="Diastolic Blood Pressure (the bottom number). *Normal Range:* 60-80 mm Hg. *To Measure:* Use a home blood pressure monitor.")
        Insulin = st.text_input('Insulin', help="2-Hour serum insulin level. *Normal Range:* < 25 mU/L. *To Measure:* This requires a blood test ordered by a doctor.")
        BMI = st.text_input('Body Mass Index (BMI)', help="Your body mass index. *Normal Range:* 18.5 - 24.9. *To Calculate:* Weight (kg) / [Height (m)]^2. You can calculate this at home with a scale and measuring tape.")
    
    if st.button('Predict Diabetes Risk', use_container_width=True):
        if diabetes_model and diabetes_scaler:
            # Placeholder for prediction logic
            pass 
        else:
            st.warning("Model not loaded. Please check your model files.")

if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction")
    st.markdown("Enter the values below. Hover over the *(?)* for detailed information on each parameter.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', help="Your age in years.")
        trestbps = st.text_input('Resting Blood Pressure', help="Systolic blood pressure (top number) in mm Hg. *Normal Range:* 90-120 mm Hg. *To Measure:* Use a home blood pressure monitor.")
        restecg = st.selectbox('Resting ECG', [0,1,2], format_func=lambda x: {0:"Normal", 1:"ST-T Abnormality", 2:"Hypertrophy"}[x], help="Results of a resting electrocardiogram. *Standard Value:* 0 (Normal). *To Measure:* Requires an ECG test at a clinic.")
        oldpeak = st.text_input('ST Depression', help="ST depression induced by exercise. *Normal Value:* 0. *To Measure:* Measured during a stress test by a medical professional.")
    with col2:
        sex = st.selectbox('Sex', [0,1], format_func=lambda x: "Female" if x==0 else "Male", help="Your biological sex.")
        chol = st.text_input('Serum Cholesterol', help="Total cholesterol level. *Normal Range:* < 200 mg/dL. *To Measure:* Requires a blood test.")
        thalach = st.text_input('Max Heart Rate Achieved', help="Highest heart rate during a stress test. *To Calculate (Estimate):* 220 minus your age. *To Measure:* Recorded during a medically supervised stress test.")
        slope = st.selectbox('Slope of Peak Exercise ST', [0,1,2], format_func=lambda x: {0:"Upsloping", 1:"Flat", 2:"Downsloping"}[x], help="The slope of the ST segment during peak exercise. *Standard Value:* 0 (Upsloping). *To Measure:* Recorded during a stress test.")
    with col3:
        cp = st.selectbox('Chest Pain Type', [0,1,2,3], format_func=lambda x: {0:"Typical Angina", 1:"Atypical Angina", 2:"Non-anginal", 3:"Asymptomatic"}[x], help="The type of chest pain experienced.")
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0,1], format_func=lambda x: "False" if x==0 else "True", help="Whether your fasting blood sugar is over 120 mg/dL. *Normal:* False (0). *To Measure:* Requires a blood test.")
        exang = st.selectbox('Exercise Induced Angina', [0,1], format_func=lambda x: "No" if x==0 else "Yes", help="Whether you experience chest pain during exercise. *Normal:* No (0).")
        ca = st.text_input('Major Vessels Colored', help="Number of major vessels (0-3) colored by flourosopy. *Normal:* 0. *To Measure:* A specialized imaging test.")
    
    thal = st.selectbox('Thalassemia Defect', [1,2,3], format_func=lambda x: {1:"Normal", 2:"Fixed Defect", 3:"Reversible Defect"}[x], help="A blood disorder. *Normal:* 1. *To Measure:* Requires specific blood tests.")
    
    if st.button('Predict Heart Disease Risk', use_container_width=True):
        if heart_model and heart_scaler:
            # Placeholder for prediction logic
            pass
        else:
             st.warning("Model not loaded. Please check your model files.")

if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction")
    st.markdown("Enter vocal measurements below. These are typically measured by specialized software. Hover over *(?)* for info.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        MDVP_Fo = st.text_input('MDVP:Fo(Hz)', help="Average vocal fundamental frequency. *Normal Range (Male):* 85-180Hz. *Normal Range (Female):* 165-255Hz.")
        MDVP_Jitter_percent = st.text_input('MDVP:Jitter(%)', help="Measure of frequency variation. *Standard:* < 1%.")
        MDVP_Shimmer = st.text_input('MDVP:Shimmer', help="Measure of amplitude variation. *Standard:* Low values are healthier.")
        HNR = st.text_input('HNR', help="Harmonics-to-Noise Ratio. Higher is better. *Standard:* > 20.")
    with col2:
        MDVP_Fhi = st.text_input('MDVP:Fhi(Hz)', help="Maximum vocal fundamental frequency.")
        MDVP_Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', help="Absolute jitter.")
        MDVP_Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', help="Shimmer in decibels.")
        RPDE = st.text_input('RPDE', help="Recurrence Period Density Entropy.")
    with col3:
        MDVP_Flo = st.text_input('MDVP:Flo(Hz)', help="Minimum vocal fundamental frequency.")
        MDVP_RAP = st.text_input('MDVP:RAP', help="Relative Average Perturbation.")
        Shimmer_APQ3 = st.text_input('Shimmer:APQ3', help="Three-point Amplitude Perturbation Quotient.")
        spread1 = st.text_input('spread1', help="Nonlinear fundamental frequency variation.")
        D2 = st.text_input('D2', help="Correlation dimension.")
    with col4:
        MDVP_PPQ = st.text_input('MDVP:PPQ', help="Five-point Period Perturbation Quotient.")
        Jitter_DDP = st.text_input('Jitter:DDP', help="Jitter difference of differences.")
        Shimmer_APQ5 = st.text_input('Shimmer:APQ5', help="Five-point Amplitude Perturbation Quotient.")
        spread2 = st.text_input('spread2', help="Nonlinear fundamental frequency variation.")
        PPE = st.text_input('PPE', help="Pitch Period Entropy.")

    if st.button("Predict Parkinson's Risk", use_container_width=True):
        if parkinsons_model and parkinsons_scaler:
            # Placeholder for prediction logic
            pass
        else:
             st.warning("Model not loaded. Please check your model files.")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from textblob import TextBlob
import time
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Workplace AI", page_icon="🌌", layout="centered")

# --- 2. CSS FIX & STYLING ---
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 0rem;}
    .stApp { background-color: #1a1b26; color: #a9b1d6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .glass-card { background: rgba(36, 40, 59, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(65, 72, 104, 0.5); padding: 25px; border-radius: 16px; margin-top: 15px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); }
    .stButton>button { background: linear-gradient(135deg, #7aa2f7 0%, #bb9af7 100%); color: #1a1b26; font-weight: 800; font-size: 1.1em; padding: 12px; border-radius: 12px; border: none; width: 100%; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(187, 154, 247, 0.6); }
    .logo-footer { background-color: #16161e; padding: 30px 20px; margin-top: 50px; border-top: 1px solid #414868; border-radius: 16px 16px 0 0; }
    .social-icons { display: flex; justify-content: center; align-items: center; gap: 25px; margin-bottom: 15px; }
    .social-icons a { color: #7aa2f7; font-size: 26px; text-decoration: none; transition: 0.3s; }
    .social-icons a:hover { color: #bb9af7; transform: scale(1.2); }
    .footer-text { text-align: center; color: #565f89; font-size: 0.9em; line-height: 1.6; }
    .header-logo { font-size: 45px; color: #bb9af7; margin-bottom: 10px; display: block; text-align: center; }
    
    /* Animation for the Support Box */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SMART FILE LOCATOR & AI ENGINE ---
@st.cache_resource
def load_ml_engine():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        'train.csv', 
        os.path.join(current_dir, 'data/train.csv'), 
        os.path.join(current_dir, '..', 'data', 'train.csv') 
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
            
    if dataset_path is None:
        return None, None, "❌ Could not find 'train.csv'. Please put it in your 'ui' folder or your 'data' folder."

    try:
        df = pd.read_csv(dataset_path)
        X = pd.get_dummies(df.drop(columns=['Employee_Id', 'Stress_Level', 'Working_State'], errors='ignore'))
        y = df['Stress_Level'].map({1: 0, 2: 0, 3: 1, 4: 2, 5: 2})
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        svm = SVC(kernel='rbf', probability=True).fit(X_scaled, y)
        return X.columns, scaler, svm
    except Exception as e:
        return None, None, str(e)

feat_cols, scaler, svm_model = load_ml_engine()

# --- 4. HEADER ---
st.markdown("""
    <div>
        <i class="fas fa-brain header-logo"></i>
        <h1 style="color: #e0af68; font-weight: bold; text-align: center; margin-bottom: 5px;">Workplace Wellness AI</h1>
        <p style="color: #7aa2f7; font-size: 1.1em; text-align: center; margin-top: 0;">Intelligent diagnostic tool for employee productivity.</p>
    </div>
""", unsafe_allow_html=True)

if svm_model is None or isinstance(svm_model, str):
    st.error(svm_model)
    st.stop()

# --- 5. UI TABS ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📊 Behavioral Metrics", "💬 Qualitative Feedback"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        hrs = st.slider("Daily Working Hours", 4, 16, 9)
        pressure = st.slider("Work Pressure (1-5)", 1, 5, 3)
        support = st.slider("Manager Support (1-5)", 1, 5, 3)
    with col2:
        sleep = st.slider("Sleep Quality (1-5)", 1, 5, 3)
        satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 3)
        exercise = st.slider("Exercise Habit (1-5)", 1, 5, 3)

with tab2:
    st.markdown("<p style='color:#a9b1d6;'>Express how you are feeling about your workload.</p>", unsafe_allow_html=True)
    user_feedback = st.text_area("", "I am performing well, but the upcoming deadlines are slightly overwhelming.", height=120)

st.markdown('</div>', unsafe_allow_html=True)

st.write("")
predict_btn = st.button("Generate Diagnostic Report")

# --- 6. POP-UP RESULTS & CONDITIONAL SUPPORT ---
if predict_btn:
    user_input = pd.DataFrame(np.zeros((1, len(feat_cols))), columns=feat_cols)
    user_input['Avg_Working_Hours_Per_Day'] = hrs; user_input['Work_Pressure'] = pressure
    user_input['Manager_Support'] = support; user_input['Sleeping_Habit'] = sleep
    user_input['Job_Satisfaction'] = satisfaction; user_input['Exercise_Habit'] = exercise
    
    u_scaled = scaler.transform(user_input)
    res = svm_model.predict(u_scaled)[0]
    
    try:
        sentiment = TextBlob(user_feedback).sentiment.polarity
        s_label = "Positive 🟢" if sentiment > 0.1 else "Negative 🔴" if sentiment < -0.1 else "Neutral 🟡"
    except:
        s_label = "Neutral (Text analysis unavailable)"

    with st.spinner('Synchronizing AI Models...'):
        time.sleep(1)
        
    st.toast("✅ Analysis Complete!", icon="🧠")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Diagnostic Results")
    
    # Conditional Pop-up Style Alerts
    if res == 0:
        st.success("🟢 **LOW STRESS RISK:** Your behavioral metrics indicate optimal efficiency and a healthy work-life balance.")
        st.balloons()
    elif res == 1:
        st.warning("🟡 **MODERATE STRESS RISK:** You are showing early signs of burnout. Consider managing your workload and taking short breaks.")
    else:
        st.error("🔴 **HIGH STRESS RISK:** Critical stress markers detected! Immediate attention to your workload and well-being is highly recommended.")
        
        # --- NEW: POP-UP STYLE HELPLINE BOX FOR HIGH RISK ---
        st.toast("🆘 Support resources unlocked below.", icon="🚨")
        st.markdown("""
            <div style="background-color: rgba(217, 48, 37, 0.1); border-left: 5px solid #f7768e; padding: 20px; border-radius: 8px; margin-top: 15px; animation: slideIn 0.8s ease-out;">
                <h4 style="color: #f7768e; margin-top: 0; margin-bottom: 10px;"><i class="fas fa-life-ring"></i> Immediate Support Resources</h4>
                <p style="color: #a9b1d6; font-size: 0.95em; margin-bottom: 15px;">Your well-being is our priority. Please reach out to the following confidential resources for support:</p>
                <ul style="list-style-type: none; padding-left: 0; line-height: 1.8;">
                    <li>📞 <b>Kiran Mental Health Helpline:</b> <a href="tel:18005990019" style="color: #7aa2f7; text-decoration: none; font-weight: bold;">1800-599-0019</a></li>
                    <li>📞 <b>Vandrevala Foundation:</b> <a href="tel:9999666555" style="color: #7aa2f7; text-decoration: none; font-weight: bold;">9999 666 555</a></li>
                    <li>📧 <b>Corporate HR Wellness Team:</b> <a href="mailto:wellness@usar.edu" style="color: #7aa2f7; text-decoration: none; font-weight: bold;">wellness@usar.edu</a></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. FOOTER ---
st.write("")
with st.expander("🛡️ Privacy Policy & System Information"):
    st.markdown("**Session Security:** Data inputted here is processed securely and is strictly temporary. We do not retain behavioral metrics or journal text.\n**Model Boundaries:** This tool acts as an early warning system based on statistical behavior, not as a diagnostic medical tool.")

st.markdown("""
    <div class="logo-footer">
        <div class="social-icons">
            <a href="mailto:vk8964210@gmail.com" target="_blank" title="Email"><i class="fas fa-envelope"></i></a>
            <a href="https://linkedin.com/in/vinay-kumar0805" target="_blank" title="LinkedIn"><i class="fab fa-linkedin"></i></a>
            <a href="https://github.com/vinaysingh-05" target="_blank" title="GitHub"><i class="fab fa-github"></i></a>
            <a href="#" target="_blank" title="University Information"><i class="fas fa-university"></i></a>
        </div>
        <div class="footer-text"> | © 2026</i>
        </div>
    </div>
""", unsafe_allow_html=True)
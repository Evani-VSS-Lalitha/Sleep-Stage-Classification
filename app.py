import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import signal
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Sleep Analysis",
    page_icon="✨",
    layout="wide",
)

# --- Load Model and Assets ---
@st.cache_resource
def load_assets():
    """Loads the model and creates the SHAP explainer."""
    try:
        model = joblib.load('sleep_final_model.pkl')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except FileNotFoundError:
        return None, None

model, explainer = load_assets()
class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
class_map = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
feature_names = [
    'EEG1-Delta', 'EEG1-Theta', 'EEG1-Alpha', 'EEG1-Beta',
    'EEG2-Delta', 'EEG2-Theta', 'EEG2-Alpha', 'EEG2-Beta',
    'EMG-Variance', 'Resp-Variance', 'Temp-Mean'
]

# --- Helper Functions (No changes here) ---
def calculate_sleep_statistics(predictions):
    stats = {}
    total_epochs = len(predictions)
    epoch_duration_mins = 0.5
    for i, name in class_map.items():
        count = np.count_nonzero(predictions == i)
        stats[name] = {'mins': count * epoch_duration_mins, 'percent': (count / total_epochs) * 100 if total_epochs > 0 else 0}
    sleep_epochs = np.count_nonzero(predictions != 0)
    stats['Total Sleep Time'] = sleep_epochs * epoch_duration_mins
    time_in_bed = total_epochs * epoch_duration_mins
    stats['Sleep Efficiency'] = (stats['Total Sleep Time'] / time_in_bed) * 100 if time_in_bed > 0 else 0
    return stats

def plot_hypnogram(predictions):
    stage_levels = {'Wake': 5, 'N1': 4, 'N2': 3, 'REM': 2, 'N3': 1}
    y_labels = [class_map.get(p, 'Unknown') for p in predictions]
    y_values = [stage_levels.get(label, 0) for label in y_labels]
    x_time = np.arange(len(predictions)) * 0.5
    df = pd.DataFrame({'Time (minutes)': x_time, 'Sleep Stage': y_labels, 'Level': y_values})
    fig = px.line(df, x='Time (minutes)', y='Level', title='Sleep Hypnogram')
    fig.update_yaxes(tickvals=list(stage_levels.values()), ticktext=list(stage_levels.keys()))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- App Layout ---
st.title('✨ Advanced Sleep Analysis')
st.markdown("An interactive tool to analyze sleep data and understand the model's decisions.")

if model is None:
    st.error("Model file (`sleep_final_model.pkl`) not found. Please ensure it's in the same folder and you have run the final training script.")
else:
    tab1, tab2 = st.tabs(["📊 Full Night Analysis", "🔬 Epoch Explorer"])

    with tab1:
        st.header("Generate a Full Sleep Report")
        age = st.number_input("Enter Age for Personalized Benchmarking", min_value=18, max_value=100, value=35, step=1)
        uploaded_features_file = st.file_uploader("Upload your 11-feature file", type="npy", key="features_upload")
        
        if uploaded_features_file:
            X_features = np.load(uploaded_features_file)
            st.session_state['X_features'] = X_features
            predictions = model.predict(X_features)
            stats = calculate_sleep_statistics(predictions)
            
            st.divider()
            st.subheader("Sleep Report")
            col1, col2 = st.columns([1, 2])
            with col1:
                col1.metric("Total Sleep Time", f"{stats['Total Sleep Time']:.0f} mins")
                col1.metric("Sleep Efficiency", f"{stats['Sleep Efficiency']:.1f}%")
                
                st.subheader("Sleep Architecture vs. Typical Norms")
                if age < 30: norms = {'Wake':'<5%','N1':'2-5%','N2':'45-55%','N3':'20-25%','REM':'20-25%'}
                elif age < 60: norms = {'Wake':'5-10%','N1':'3-7%','N2':'50-60%','N3':'13-23%','REM':'18-23%'}
                else: norms = {'Wake':'>10%','N1':'5-10%','N2':'50-60%','N3':'5-15%','REM':'15-20%'}

                benchmark_df = pd.DataFrame({
                    'Stage': class_names,
                    'Your Night (%)': [stats[s]['percent'] for s in class_names],
                    f'Norms for Age {age} (%)': [norms[s] for s in class_names]
                }).set_index('Stage')
                st.table(benchmark_df.style.format({'Your Night (%)': '{:.1f}'}))
                
                csv = convert_df_to_csv(benchmark_df.reset_index())
                st.download_button(label="📥 Download Report as CSV", data=csv,
                                   file_name=f'sleep_report_{uploaded_features_file.name.replace(".npy", "")}.csv',
                                   mime='text/csv')

            with col2:
                 st.plotly_chart(plot_hypnogram(predictions), use_container_width=True)

    with tab2:
        st.header("Explore a Single Epoch and Explain the Prediction")
        st.subheader("1. Visualize Raw Signals")
        uploaded_raw_file = st.file_uploader("Upload the raw 5-channel data", type="npy", key="raw_upload")
        
        if uploaded_raw_file:
            X_raw = np.load(uploaded_raw_file)
            epoch_index = st.slider("Select an epoch to visualize:", 0, len(X_raw) - 1, 0)
            epoch_data = X_raw[epoch_index]
            
            fig_raw = go.Figure()
            channel_names = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EMG', 'Respiration', 'Temperature']
            for i, name in enumerate(channel_names):
                fig_raw.add_trace(go.Scatter(y=epoch_data[i], name=name))
            st.plotly_chart(fig_raw, use_container_width=True)
            
            st.divider()
            st.subheader("2. Explain the Model's Prediction for this Epoch")
            
            if 'X_features' in st.session_state:
                X_features = st.session_state['X_features']
                if epoch_index < len(X_features):
                    single_epoch_features = X_features[epoch_index]
                    prediction = model.predict(single_epoch_features.reshape(1, -1))[0]
                    st.info(f"The model's prediction for Epoch #{epoch_index} is: **{class_map[prediction]}**")

                    shap_values_obj = explainer(single_epoch_features)
                    
                    base_value_for_class = explainer.expected_value[prediction]
                    shap_values_for_class = shap_values_obj.values[:, prediction]
                    
                    waterfall_explanation = shap.Explanation(
                        values=shap_values_for_class,
                        base_values=base_value_for_class,
                        data=single_epoch_features,
                        feature_names=feature_names
                    )
                    
                    st.write("**How each feature influenced this specific decision:**")
                    shap.plots.waterfall(waterfall_explanation, max_display=11, show=False)
                    
                    # --- THIS IS THE FIX ---
                    # Instead of passing the variable, pass plt.gcf() (get current figure)
                    st.pyplot(plt.gcf(), bbox_inches='tight')
                    plt.clf()
                else:
                    st.error("Error: The selected epoch index is out of bounds for the loaded feature file.")
            else:
                st.warning("To see the SHAP explanation, please upload the corresponding feature file in the 'Full Night Analysis' tab first.")
import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="UrbanFlow | Command Center", page_icon="📊", layout="wide")

# --- CUSTOM PROFESSIONAL CSS ---
st.markdown("""
    <style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background-color: #0f172a; /* Slate 900 */
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b; /* Slate 800 */
        border-right: 1px solid #334155;
    }
    
    /* KPI Cards */
    .kpi-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .kpi-title { color: #94a3b8; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px;}
    .kpi-value { color: #f8fafc; font-size: 2rem; font-weight: 700; margin: 0; }
    .kpi-value-green { color: #10b981; font-size: 2rem; font-weight: 700; margin: 0; }
    .kpi-value-blue { color: #3b82f6; font-size: 2rem; font-weight: 700; margin: 0; }

    /* Main Recommendation Panel */
    .rec-panel {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.1);
    }
    .rec-header { color: #3b82f6; font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
    .rec-zone { color: #f8fafc; font-size: 2.5rem; font-weight: 700; margin-top: 10px; margin-bottom: 20px; }
    
    /* Fix raw HTML issue by not using newlines inside f-strings */
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    try:
        zones = pd.read_csv("taxi_zone_lookup.csv")
        zone_dict = dict(zip(zones['LocationID'], zones['Zone']))
    except:
        zone_dict = {}
    return zone_dict

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("xgb_with_weather.json")
    return model

zone_dict = load_data()
model = load_model()

# --- HEADER ---
st.markdown("<h1 style='color: #f8fafc; font-size: 2.5rem; margin-bottom: 0;'>UrbanFlow Command Center</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-top: 5px;'>Prescriptive Analytics & Real-Time Fleet Optimization</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-color: #334155; margin-bottom: 30px;'>", unsafe_allow_html=True)

# --- SIDEBAR INPUTS ---
st.sidebar.markdown("<h3 style='color: #f8fafc;'>⚙️ Environment Variables</h3>", unsafe_allow_html=True)
hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 18)
temp = st.sidebar.slider("Temperature (°F)", 10, 100, 45)
precip = st.sidebar.slider("Precipitation (inches)", 0.0, 2.0, 0.0)

st.sidebar.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Powered by XGBoost<br>Dataset: 50M+ TLC Trips (2024)</p>", unsafe_allow_html=True)

execute = st.sidebar.button("🚀 Run Optimization Model", use_container_width=True)

# --- MAIN PREDICTION LOGIC ---
if execute:
    zones_to_check = [132, 237, 161, 236, 162, 138, 230, 142, 163, 186]
    results = []
    
    for zone in zones_to_check:
        input_data = pd.DataFrame({
            'zone_id': [zone], 'hour': [hour], 'day_of_week': [2], 'month': [10],
            'is_weekend': [0], 'lag_3h': [40.0], 'lag_6h': [35.0], 'lag_12h': [30.0],
            'HourlyDryBulbTemperature': [temp], 'HourlyPrecipitation': [precip], 'HourlyWindSpeed': [5.0]
        })
        
        predicted_demand = model.predict(input_data)[0]
        zone_name = zone_dict.get(zone, "Unknown Zone")
        results.append({"Zone ID": zone, "Zone Name": zone_name, "Expected Rides": predicted_demand})
    
    df_results = pd.DataFrame(results).sort_values(by="Expected Rides", ascending=False).reset_index(drop=True)
    best_zone = df_results.iloc[0]
    
    expected_rides = int(best_zone['Expected Rides'])
    est_revenue = expected_rides * 15
    
    # --- KPI ROW (Top Dashboard Metrics) ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f'<div class="kpi-card"><p class="kpi-title">System Status</p><p class="kpi-value-green">Online</p></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card"><p class="kpi-title">Current Weather</p><p class="kpi-value-blue">{temp}°F</p></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="kpi-card"><p class="kpi-title">Max Expected Rides</p><p class="kpi-value">{expected_rides} / hr</p></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card"><p class="kpi-title">Est. Revenue Potential</p><p class="kpi-value-green">${est_revenue}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- MAIN CONTENT AREA (Two Columns) ---
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        # Professional Recommendation Card (Fixed HTML)
        rec_html = f"""
        <div class="rec-panel">
            <p class="rec-header">📍 Primary Directives</p>
            <p class="rec-zone">Zone {best_zone['Zone ID']} - {best_zone['Zone Name']}</p>
            <p style="color: #94a3b8; font-size: 1rem;">Initiate repositioning to this sector immediately to capitalize on projected demand surge.</p>
        </div>
        """
        st.markdown(rec_html, unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #f8fafc; margin-top: 25px;'>Secondary Targets</h4>", unsafe_allow_html=True)
        for i in range(1, 4): # Show zones 2, 3, 4
            row = df_results.iloc[i]
            st.markdown(f"<div style='padding: 10px 0; border-bottom: 1px solid #334155;'><b style='color: #cbd5e1;'>Zone {row['Zone ID']}</b> - {row['Zone Name']} <span style='float: right; color: #3b82f6; font-weight: bold;'>{int(row['Expected Rides'])} rides</span></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h4 style='color: #f8fafc; margin-top: 0;'>Demand Forecast Distribution</h4>", unsafe_allow_html=True)
        
        # Professional Plotly Chart
        fig = px.bar(
            df_results.head(5), 
            x='Expected Rides', 
            y='Zone Name', 
            orientation='h',
            color='Expected Rides',
            color_continuous_scale=['#1e3a8a', '#3b82f6'], # Corporate blues
            text='Expected Rides'
        )
        
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family="Inter"),
            xaxis=dict(showgrid=True, gridcolor='#334155', zeroline=False, title="Predicted Rides per Hour"),
            yaxis_title=None,
            coloraxis_showscale=False
        )
        
        fig.update_xaxes(range=[0, df_results['Expected Rides'].max() * 1.15])
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside', marker_line_width=0)
        
        st.plotly_chart(fig, use_container_width=True)

else:
    # Standby State
    st.markdown("""
    <div style="background-color: #1e293b; border: 1px dashed #475569; border-radius: 8px; padding: 40px; text-align: center; margin-top: 20px;">
        <h3 style="color: #cbd5e1; margin-bottom: 10px;">System Standby</h3>
        <p style="color: #94a3b8;">Adjust the environment variables in the sidebar and click <b style="color: #3b82f6;">Run Optimization Model</b> to execute the XGBoost prediction engine.</p>
    </div>
    """, unsafe_allow_html=True)
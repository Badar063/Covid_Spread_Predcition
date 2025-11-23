import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="COVID-19 Spread Prediction Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class COVIDDashboard:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.setup_data()
    
    def setup_data(self):
        """Prepare data for visualization"""
        # Calculate additional metrics
        self.data['prediction_abs_error'] = abs(self.data['prediction_error'])
        self.data['data_type'] = self.data['data_quality'].apply(
            lambda x: 'High Quality' if pd.notna(x) and 'high' in x.lower() else 'Low Quality'
        )
        
        # Performance metrics
        self.metrics = {
            'mse': mean_squared_error(self.data['true_infection_rate'], self.data['predicted_infection_rate']),
            'mae': mean_absolute_error(self.data['true_infection_rate'], self.data['predicted_infection_rate']),
            'r2': r2_score(self.data['true_infection_rate'], self.data['predicted_infection_rate'])
        }
    
    def create_overview_metrics(self):
        """Create overview metrics cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Areas",
                f"{len(self.data):,}",
                help="Number of geographical areas analyzed"
            )
        
        with col2:
            avg_true_rate = self.data['true_infection_rate'].mean()
            st.metric(
                "Avg True Infection Rate",
                f"{avg_true_rate:.2f}%",
                help="Average true infection rate across all areas"
            )
        
        with col3:
            avg_pred_error = self.data['prediction_abs_error'].mean()
            st.metric(
                "Avg Prediction Error",
                f"{avg_pred_error:.2f}%",
                delta=f"{-avg_pred_error:.2f}%" if avg_pred_error < 5 else None,
                help="Average absolute prediction error"
            )
        
        with col4:
            st.metric(
                "Model R² Score",
                f"{self.metrics['r2']:.3f}",
                help="R-squared score of the integrated model"
            )
    
    def create_spatial_map(self):
        """Create interactive spatial map"""
        st.markdown('<div class="section-header">Spatial Distribution of Infection Rates</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            map_type = st.selectbox(
                "Select Map Type",
                ["True Infection Rates", "Predicted Infection Rates", "Prediction Error"],
                key="map_type"
            )
        
        # Create base map
        center_lat, center_lon = self.data['latitude'].mean(), self.data['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Color mapping based on selection
        if map_type == "True Infection Rates":
            values = self.data['true_infection_rate']
            cmap = 'RdYlGn_r'
            legend_name = "True Infection Rate (%)"
        elif map_type == "Predicted Infection Rates":
            values = self.data['predicted_infection_rate']
            cmap = 'RdYlGn_r'
            legend_name = "Predicted Infection Rate (%)"
        else:
            values = self.data['prediction_error']
            cmap = 'RdBu_r'
            legend_name = "Prediction Error (%)"
        
        # Normalize values for color mapping
        vmin, vmax = values.min(), values.max()
        
        for idx, row in self.data.iterrows():
            if map_type == "Prediction Error":
                # For error, use diverging colormap
                error = row['prediction_error']
                if error > 0:
                    color = f"#{min(255, int(255*(error/vmax))):02x}0000"  # Red scale
                else:
                    color = f"#0000{min(255, int(255*(abs(error)/abs(vmin))):02x}"  # Blue scale
            else:
                # For rates, use sequential colormap
                rate_value = row['true_infection_rate'] if map_type == "True Infection Rates" else row['predicted_infection_rate']
                norm_value = (rate_value - vmin) / (vmax - vmin)
                color = f"#{int(255*norm_value):02x}{int(255*(1-norm_value)):02x}00"
            
            # Add circle marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=f"""
                <b>Area:</b> {row['area_id']}<br>
                <b>True Rate:</b> {row['true_infection_rate']:.2f}%<br>
                <b>Predicted:</b> {row['predicted_infection_rate']:.2f}%<br>
                <b>Error:</b> {row['prediction_error']:.2f}%<br>
                <b>Population Density:</b> {row['population_density']:.0f}<br>
                <b>Income Level:</b> ${row['income_level']:.0f}
                """,
                color=color,
                fill=True,
                fillColor=color
            ).add_to(m)
        
        folium_static(m, width=1200, height=500)
    
    def create_performance_dashboard(self):
        """Create model performance visualizations"""
        st.markdown('<div class="section-header">Model Performance Analysis</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # True vs Predicted scatter plot
            fig = px.scatter(
                self.data,
                x='true_infection_rate',
                y='predicted_infection_rate',
                color='data_type',
                title='True vs Predicted Infection Rates',
                labels={
                    'true_infection_rate': 'True Infection Rate (%)',
                    'predicted_infection_rate': 'Predicted Infection Rate (%)',
                    'data_type': 'Data Quality'
                }
                # Removed trendline parameter
            )
            # Add perfect prediction line
            max_rate = max(self.data['true_infection_rate'].max(), 
                          self.data['predicted_infection_rate'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_rate], y=[0, max_rate],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect Prediction'
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error distribution
            fig = px.histogram(
                self.data,
                x='prediction_error',
                nbins=50,
                title='Distribution of Prediction Errors',
                labels={'prediction_error': 'Prediction Error (%)'},
                color_discrete_sequence=['#ff7f0e']
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    
    def create_feature_analysis(self):
        """Analyze feature importance and relationships"""
        st.markdown('<div class="section-header">Feature Analysis</div>', 
                   unsafe_allow_html=True)
        
        # Feature selection
        features = ['population_density', 'income_level', 'public_transit_use', 
                   'elderly_population', 'essential_workers', 'urban_proximity']
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis Feature", features, key="x_feature")
        
        with col2:
            y_feature = st.selectbox("Y-axis Feature", 
                                   ['true_infection_rate', 'predicted_infection_rate', 'prediction_error'], 
                                   key="y_feature")
        
        # Create scatter plot without trendline
        fig = px.scatter(
            self.data,
            x=x_feature,
            y=y_feature,
            color='data_type',
            size='population_density',
            hover_data=['area_id'],
            title=f'{y_feature.replace("_", " ").title()} vs {x_feature.replace("_", " ").title()}'
            # Removed trendline parameter
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def create_bias_analysis(self):
        """Analyze prediction biases"""
        st.markdown('<div class="section-header">Bias Analysis</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bias by income level
            fig = px.scatter(
                self.data,
                x='income_level',
                y='prediction_error',
                color='data_type',
                title='Prediction Error vs Income Level',
                labels={
                    'income_level': 'Income Level ($)',
                    'prediction_error': 'Prediction Error (%)'
                }
                # Removed trendline parameter
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bias by urban proximity
            fig = px.scatter(
                self.data,
                x='urban_proximity',
                y='prediction_error',
                color='data_type',
                title='Prediction Error vs Urban Proximity',
                labels={
                    'urban_proximity': 'Urban Proximity (0-1)',
                    'prediction_error': 'Prediction Error (%)'
                }
                # Removed trendline parameter
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    
    def create_uncertainty_analysis(self):
        """Analyze prediction uncertainty"""
        st.markdown('<div class="section-header">Uncertainty Analysis</div>', 
                   unsafe_allow_html=True)
        
        self.data['uncertainty_level'] = self.data['prediction_abs_error'] / self.data['true_infection_rate']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                self.data,
                x='data_type',
                y='prediction_abs_error',
                title='Prediction Error by Data Quality',
                labels={
                    'data_type': 'Data Quality',
                    'prediction_abs_error': 'Absolute Prediction Error (%)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            high_uncertainty = self.data.nlargest(10, 'prediction_abs_error')
            fig = px.bar(
                high_uncertainty,
                x='area_id',
                y='prediction_abs_error',
                title='Top 10 Areas with Highest Prediction Errors',
                labels={
                    'area_id': 'Area ID',
                    'prediction_abs_error': 'Absolute Error (%)'
                },
                color='prediction_abs_error',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def create_comparison_metrics(self):
        """Create detailed performance metrics"""
        st.markdown('<div class="section-header">Detailed Performance Metrics</div>', 
                   unsafe_allow_html=True)
        
        metrics_by_quality = []
        for quality in self.data['data_type'].unique():
            subset = self.data[self.data['data_type'] == quality]
            metrics_by_quality.append({
                'Data Quality': quality,
                'Count': len(subset),
                'Avg True Rate': subset['true_infection_rate'].mean(),
                'Avg Predicted Rate': subset['predicted_infection_rate'].mean(),
                'Avg Error': subset['prediction_error'].mean(),
                'Avg Abs Error': subset['prediction_abs_error'].mean(),
                'RMSE': np.sqrt(mean_squared_error(subset['true_infection_rate'], 
                                                 subset['predicted_infection_rate']))
            })
        
        metrics_df = pd.DataFrame(metrics_by_quality)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(metrics_df.round(3), use_container_width=True)
        
        with col2:
            fig = px.violin(
                self.data,
                x='data_type',
                y='prediction_abs_error',
                color='data_type',
                title='Distribution of Absolute Errors by Data Quality',
                labels={
                    'data_type': 'Data Quality',
                    'prediction_abs_error': 'Absolute Prediction Error (%)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def run_dashboard(self):
        """Run the complete dashboard"""
        st.markdown('<h1 class="main-header">🦠 COVID-19 Spread Prediction Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        self.create_overview_metrics()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌍 Spatial Analysis", 
            "📊 Performance", 
            "🔍 Features", 
            "⚖️ Bias Analysis",
            "📈 Metrics"
        ])
        
        with tab1:
            self.create_spatial_map()
        
        with tab2:
            self.create_performance_dashboard()
        
        with tab3:
            self.create_feature_analysis()
        
        with tab4:
            self.create_bias_analysis()
            self.create_uncertainty_analysis()
        
        with tab5:
            self.create_comparison_metrics()
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>COVID-19 Integrated Prediction Model Dashboard</p>
            <p>Combining self-reported data and testing data for accurate spread prediction</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    try:
        dashboard = COVIDDashboard('covid_integrated_predictions.csv')
        dashboard.run_dashboard()
    except FileNotFoundError:
        st.error("❌ Data file 'covid_integrated_predictions.csv' not found. Please ensure it's in the same directory.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

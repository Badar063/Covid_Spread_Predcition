import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1rem;
        padding: 10px 16px;
    }
</style>
""", unsafe_allow_html=True)

class COVIDDashboard:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.setup_data()
    
    def setup_data(self):
        """Prepare data for visualization"""
        # Clean the data
        self.data = self.data.fillna(0)
        
        # Calculate additional metrics
        self.data['prediction_abs_error'] = abs(self.data['prediction_error'])
        self.data['data_type'] = 'Low Quality'  # Default
        if 'data_quality' in self.data.columns:
            self.data['data_type'] = self.data['data_quality'].apply(
                lambda x: 'High Quality' if pd.notna(x) and 'high' in str(x).lower() else 'Low Quality'
            )
        
        # Performance metrics
        self.metrics = {
            'mse': mean_squared_error(self.data['true_infection_rate'], self.data['predicted_infection_rate']),
            'mae': mean_absolute_error(self.data['true_infection_rate'], self.data['predicted_infection_rate']),
            'r2': r2_score(self.data['true_infection_rate'], self.data['predicted_infection_rate']),
            'rmse': np.sqrt(mean_squared_error(self.data['true_infection_rate'], self.data['predicted_infection_rate']))
        }
    
    def create_overview_metrics(self):
        """Create overview metrics cards"""
        st.markdown('<div class="section-header">📊 Dashboard Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Total Areas Analyzed",
                f"{len(self.data):,}",
                help="Number of geographical areas in the dataset"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_true_rate = self.data['true_infection_rate'].mean()
            st.metric(
                "Average True Infection Rate",
                f"{avg_true_rate:.2f}%",
                help="Average true infection rate across all areas"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_pred_error = self.data['prediction_abs_error'].mean()
            st.metric(
                "Average Prediction Error",
                f"{avg_pred_error:.2f}%",
                delta=f"{-avg_pred_error:.2f}%" if avg_pred_error < 5 else None,
                help="Average absolute prediction error"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Model R² Score",
                f"{self.metrics['r2']:.3f}",
                help="R-squared score indicating model performance"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    def create_spatial_map(self):
        """Create interactive spatial map"""
        st.markdown('<div class="section-header">🌍 Spatial Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            map_type = st.selectbox(
                "Select Visualization",
                ["True Infection Rates", "Predicted Infection Rates", "Prediction Error", "Population Density"],
                key="map_type"
            )
            
            st.info("💡 Click on markers to see detailed area information")
        
        with col2:
            # Create base map
            center_lat, center_lon = self.data['latitude'].mean(), self.data['longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
            
            # Determine values for coloring
            if map_type == "True Infection Rates":
                values = self.data['true_infection_rate']
                column = 'true_infection_rate'
                color_scheme = 'RdYlGn_r'
            elif map_type == "Predicted Infection Rates":
                values = self.data['predicted_infection_rate']
                column = 'predicted_infection_rate'
                color_scheme = 'RdYlGn_r'
            elif map_type == "Prediction Error":
                values = self.data['prediction_error']
                column = 'prediction_error'
                color_scheme = 'RdBu_r'
            else:
                values = self.data['population_density']
                column = 'population_density'
                color_scheme = 'Blues'
            
            # Normalize values for circle sizes
            vmin, vmax = values.min(), values.max()
            size_range = (8, 20)
            
            for idx, row in self.data.iterrows():
                # Calculate normalized size
                if vmax > vmin:
                    normalized_value = (row[column] - vmin) / (vmax - vmin)
                    size = size_range[0] + normalized_value * (size_range[1] - size_range[0])
                else:
                    size = size_range[0]
                
                # Determine color based on map type
                if map_type == "Prediction Error":
                    if row['prediction_error'] > 0:
                        color = 'red'  # Overestimation
                    else:
                        color = 'blue'  # Underestimation
                else:
                    color = 'green'
                
                # Add circle marker
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=size,
                    popup=folium.Popup(f"""
                    <b>Area ID:</b> {row['area_id']}<br>
                    <b>True Rate:</b> {row['true_infection_rate']:.2f}%<br>
                    <b>Predicted Rate:</b> {row['predicted_infection_rate']:.2f}%<br>
                    <b>Prediction Error:</b> {row['prediction_error']:.2f}%<br>
                    <b>Population Density:</b> {row['population_density']:,.0f}/km²<br>
                    <b>Income Level:</b> ${row['income_level']:,.0f}<br>
                    <b>Elderly Population:</b> {row['elderly_population']:.1f}%
                    """, max_width=300),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6
                ).add_to(m)
            
            folium_static(m, width=1000, height=600)
    
    def create_performance_analysis(self):
        """Create model performance visualizations"""
        st.markdown('<div class="section-header">📈 Model Performance Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # True vs Predicted scatter plot
            fig = px.scatter(
                self.data,
                x='true_infection_rate',
                y='predicted_infection_rate',
                color='data_type',
                size='population_density',
                title='True vs Predicted Infection Rates',
                labels={
                    'true_infection_rate': 'True Infection Rate (%)',
                    'predicted_infection_rate': 'Predicted Infection Rate (%)',
                    'data_type': 'Data Quality'
                },
                hover_data=['area_id', 'income_level']
            )
            
            # Add perfect prediction line
            max_rate = max(self.data['true_infection_rate'].max(), 
                          self.data['predicted_infection_rate'].max()) + 5
            fig.add_trace(go.Scatter(
                x=[0, max_rate], y=[0, max_rate],
                mode='lines',
                line=dict(dash='dash', color='gray', width=2),
                name='Perfect Prediction Line'
            ))
            
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error distribution with box plot
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=('Error Distribution', 'Error by Data Quality'))
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=self.data['prediction_error'], 
                           nbinsx=30, name='Error Distribution',
                           marker_color='#ff7f0e'),
                row=1, col=1
            )
            
            # Box plot by data quality
            for quality in self.data['data_type'].unique():
                subset = self.data[self.data['data_type'] == quality]
                fig.add_trace(
                    go.Box(y=subset['prediction_error'], name=quality,
                          marker_color='red' if quality == 'High Quality' else 'blue'),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, showlegend=False)
            fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
            st.plotly_chart(fig, use_container_width=True)
    
    def create_feature_analysis(self):
        """Analyze feature importance and relationships"""
        st.markdown('<div class="section-header">🔍 Feature Relationship Analysis</div>', unsafe_allow_html=True)
        
        # Feature selection
        features = [
            'population_density', 'income_level', 'public_transit_use', 
            'elderly_population', 'essential_workers', 'urban_proximity'
        ]
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            x_feature = st.selectbox("X-axis Feature", features, key="x_feature")
        
        with col2:
            y_feature = st.selectbox("Y-axis Feature", 
                                   ['true_infection_rate', 'predicted_infection_rate', 'prediction_error'], 
                                   key="y_feature")
        
        with col3:
            color_by = st.selectbox("Color By", 
                                  ['data_type', 'prediction_error', 'urban_proximity'],
                                  key="color_by")
        
        # Create scatter plot
        fig = px.scatter(
            self.data,
            x=x_feature,
            y=y_feature,
            color=color_by,
            size='population_density',
            hover_data=['area_id'],
            title=f'{y_feature.replace("_", " ").title()} vs {x_feature.replace("_", " ").title()}',
            color_continuous_scale='Viridis'
        )
        
        # Add correlation line if appropriate
        if color_by != 'prediction_error':
            z = np.polyfit(self.data[x_feature], self.data[y_feature], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=self.data[x_feature],
                y=p(self.data[x_feature]),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Trend Line'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.markdown("#### Correlation Heatmap")
        corr_features = features + ['true_infection_rate', 'predicted_infection_rate', 'prediction_error']
        correlation_matrix = self.data[corr_features].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title='Feature Correlation Matrix'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def create_bias_analysis(self):
        """Analyze prediction biases"""
        st.markdown('<div class="section-header">⚖️ Bias and Fairness Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bias by income quartiles
            self.data['income_quartile'] = pd.qcut(self.data['income_level'], 4, 
                                                 labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])
            
            fig = px.box(
                self.data,
                x='income_quartile',
                y='prediction_error',
                title='Prediction Error Distribution by Income Quartiles',
                labels={
                    'income_quartile': 'Income Quartile',
                    'prediction_error': 'Prediction Error (%)'
                }
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bias by urban proximity categories
            self.data['urban_category'] = pd.cut(self.data['urban_proximity'], 
                                              bins=[0, 0.33, 0.66, 1], 
                                              labels=['Rural', 'Suburban', 'Urban'])
            
            bias_summary = self.data.groupby('urban_category').agg({
                'prediction_error': ['mean', 'std', 'count']
            }).round(3)
            bias_summary.columns = ['Mean Error', 'Std Error', 'Count']
            bias_summary = bias_summary.reset_index()
            
            st.dataframe(bias_summary, use_container_width=True)
            
            fig = px.bar(
                bias_summary,
                x='urban_category',
                y='Mean Error',
                title='Average Prediction Error by Urban Category',
                labels={'urban_category': 'Urban Category', 'Mean Error': 'Average Error (%)'},
                color='Mean Error',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def create_uncertainty_analysis(self):
        """Analyze prediction uncertainty"""
        st.markdown('<div class="section-header">🎯 Uncertainty and Error Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 worst predictions
            worst_predictions = self.data.nlargest(10, 'prediction_abs_error')[[
                'area_id', 'true_infection_rate', 'predicted_infection_rate', 
                'prediction_error', 'population_density'
            ]].round(3)
            
            st.subheader("🔴 Top 10 Largest Prediction Errors")
            st.dataframe(worst_predictions, use_container_width=True)
        
        with col2:
            # Best predictions (lowest errors)
            best_predictions = self.data.nsmallest(10, 'prediction_abs_error')[[
                'area_id', 'true_infection_rate', 'predicted_infection_rate', 
                'prediction_error', 'data_type'
            ]].round(3)
            
            st.subheader("🟢 Top 10 Most Accurate Predictions")
            st.dataframe(best_predictions, use_container_width=True)
        
        # Error analysis by feature ranges
        st.subheader("Error Analysis Across Different Ranges")
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            feature = st.selectbox("Analyze errors by:", 
                                 ['population_density', 'income_level', 'elderly_population'],
                                 key="error_analysis")
        
        with analysis_col2:
            bins = st.slider("Number of bins", 3, 10, 5)
        
        with analysis_col3:
            st.metric("Overall MAE", f"{self.metrics['mae']:.3f}")
            st.metric("Overall RMSE", f"{self.metrics['rmse']:.3f}")
        
        # Create binned analysis
        self.data[f'{feature}_bins'] = pd.cut(self.data[feature], bins=bins)
        binned_errors = self.data.groupby(f'{feature}_bins').agg({
            'prediction_abs_error': 'mean',
            'true_infection_rate': 'mean',
            'area_id': 'count'
        }).round(3).reset_index()
        
        binned_errors.columns = [f'{feature}_range', 'Average Error', 'Avg True Rate', 'Area Count']
        
        fig = px.bar(
            binned_errors,
            x=f'{feature}_range',
            y='Average Error',
            title=f'Average Prediction Error by {feature.replace("_", " ").title()} Ranges',
            labels={f'{feature}_range': f'{feature.replace("_", " ").title()} Range', 'Average Error': 'Avg Abs Error (%)'},
            color='Average Error',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def run_dashboard(self):
        """Run the complete dashboard"""
        # Header
        st.markdown('<h1 class="main-header">🦠 COVID-19 Integrated Prediction Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Overview metrics
        self.create_overview_metrics()
        
        # Main content with tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌍 Spatial View", 
            "📊 Performance", 
            "🔍 Features", 
            "⚖️ Bias Analysis",
            "🎯 Error Analysis"
        ])
        
        with tab1:
            self.create_spatial_map()
        
        with tab2:
            self.create_performance_analysis()
        
        with tab3:
            self.create_feature_analysis()
        
        with tab4:
            self.create_bias_analysis()
        
        with tab5:
            self.create_uncertainty_analysis()
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; color: gray;'>
                <h4>COVID-19 Integrated Prediction Model</h4>
                <p>Combining multiple data sources for accurate infection spread prediction</p>
                <p><em>Last updated: December 2024</em></p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main function to run the dashboard"""
    try:
        # Load data
        dashboard = COVIDDashboard('covid_integrated_predictions.csv')
        
        # Sidebar
        st.sidebar.title("Navigation")
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **About this Dashboard:**
        This dashboard visualizes COVID-19 infection rate predictions using integrated modeling that combines self-reported data and official testing data.
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Dataset Info")
        st.sidebar.write(f"Total areas: {len(dashboard.data):,}")
        st.sidebar.write(f"Features: {len(dashboard.data.columns)}")
        st.sidebar.write(f"High quality data areas: {len(dashboard.data[dashboard.data['data_type'] == 'High Quality']):,}")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Performance")
        st.sidebar.metric("R² Score", f"{dashboard.metrics['r2']:.3f}")
        st.sidebar.metric("MAE", f"{dashboard.metrics['mae']:.3f}")
        st.sidebar.metric("RMSE", f"{dashboard.metrics['rmse']:.3f}")
        
        # Run dashboard
        dashboard.run_dashboard()
        
    except FileNotFoundError:
        st.error("""
        ❌ **Data file not found!**
        
        Please ensure `covid_integrated_predictions.csv` is in the same directory as this script.
        """)
    except Exception as e:
        st.error(f"""
        ❌ **An error occurred:**
        
        `{str(e)}`
        
        Please check your data file and try again.
        """)

if __name__ == "__main__":
    main()

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
        padding: 1rem;
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
</style>
""", unsafe_allow_html=True)

class COVIDDashboard:
    def __init__(self, data_path):
        try:
            self.data = pd.read_csv(data_path)
            self.clean_data()
            self.setup_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            self.data = pd.DataFrame()
    
    def clean_data(self):
        """Clean and prepare the data, removing any problematic types"""
        if self.data.empty:
            return
            
        # Fill NaN values
        self.data = self.data.fillna(0)
        
        # Convert all columns to safe types
        for col in self.data.columns:
            # Handle numeric columns
            if self.data[col].dtype == 'object':
                try:
                    self.data[col] = pd.to_numeric(self.data[col], errors='ignore')
                except:
                    pass
            
            # Convert any remaining object types to string to avoid serialization issues
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].astype(str)
    
    def setup_data(self):
        """Prepare data for visualization"""
        if self.data.empty:
            return
            
        # Calculate additional metrics
        self.data['prediction_abs_error'] = abs(self.data['prediction_error'])
        
        # Handle data quality column safely
        # Check if data_quality column exists and handle it
        if 'data_quality' in self.data.columns:
            # Convert to string first to avoid any interval objects
            self.data['data_quality'] = self.data['data_quality'].astype(str)
            self.data['data_type'] = self.data['data_quality'].apply(
                lambda x: 'High Quality' if 'high' in x.lower() else 'Low Quality'
            )
        else:
            self.data['data_type'] = 'Low Quality'
        
        # Create simplified categories for analysis (avoiding intervals)
        self.create_safe_categories()
        
        # Performance metrics
        try:
            self.metrics = {
                'mse': mean_squared_error(self.data['true_infection_rate'], self.data['predicted_infection_rate']),
                'mae': mean_absolute_error(self.data['true_infection_rate'], self.data['predicted_infection_rate']),
                'r2': r2_score(self.data['true_infection_rate'], self.data['predicted_infection_rate']),
                'rmse': np.sqrt(mean_squared_error(self.data['true_infection_rate'], self.data['predicted_infection_rate']))
            }
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            self.metrics = {'mse': 0, 'mae': 0, 'r2': 0, 'rmse': 0}
    
    def create_safe_categories(self):
        """Create categories without using pandas Interval objects"""
        # Income categories (simple quantile-based)
        if 'income_level' in self.data.columns:
            income_quantiles = self.data['income_level'].quantile([0.25, 0.5, 0.75])
            conditions = [
                self.data['income_level'] <= income_quantiles[0.25],
                (self.data['income_level'] > income_quantiles[0.25]) & (self.data['income_level'] <= income_quantiles[0.5]),
                (self.data['income_level'] > income_quantiles[0.5]) & (self.data['income_level'] <= income_quantiles[0.75]),
                self.data['income_level'] > income_quantiles[0.75]
            ]
            choices = ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
            self.data['income_category'] = np.select(conditions, choices, default='Q2')
        
        # Urban categories (simple thresholds)
        if 'urban_proximity' in self.data.columns:
            conditions = [
                self.data['urban_proximity'] <= 0.33,
                (self.data['urban_proximity'] > 0.33) & (self.data['urban_proximity'] <= 0.66),
                self.data['urban_proximity'] > 0.66
            ]
            choices = ['Rural', 'Suburban', 'Urban']
            self.data['urban_category'] = np.select(conditions, choices, default='Suburban')
    
    def create_overview_metrics(self):
        """Create overview metrics cards"""
        if self.data.empty:
            st.warning("No data available")
            return
            
        st.markdown('<div class="section-header">📊 Dashboard Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Areas", f"{len(self.data):,}")
        
        with col2:
            avg_true_rate = self.data['true_infection_rate'].mean()
            st.metric("Avg True Infection Rate", f"{avg_true_rate:.2f}%")
        
        with col3:
            avg_pred_error = self.data['prediction_abs_error'].mean()
            st.metric("Avg Prediction Error", f"{avg_pred_error:.2f}%")
        
        with col4:
            st.metric("Model R² Score", f"{self.metrics['r2']:.3f}")
    
    def create_spatial_map(self):
        """Create interactive spatial map"""
        if self.data.empty:
            st.warning("No data available for spatial map")
            return
            
        st.markdown('<div class="section-header">🌍 Spatial Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            map_type = st.selectbox(
                "Select Visualization",
                ["True Infection Rates", "Predicted Infection Rates", "Prediction Error"],
                key="map_type"
            )
            
            st.info("💡 Click on markers to see area details")
        
        with col2:
            # Create base map
            center_lat, center_lon = self.data['latitude'].mean(), self.data['longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
            
            for idx, row in self.data.iterrows():
                # Determine color and size based on map type
                if map_type == "True Infection Rates":
                    value = row['true_infection_rate']
                    color = 'red'
                    size = max(5, min(20, value / 2))
                elif map_type == "Predicted Infection Rates":
                    value = row['predicted_infection_rate']
                    color = 'blue'
                    size = max(5, min(20, value / 2))
                else:  # Prediction Error
                    value = abs(row['prediction_error'])
                    color = 'red' if row['prediction_error'] > 0 else 'blue'
                    size = max(5, min(20, value))
                
                # Create popup text safely
                popup_text = f"""
                <b>Area ID:</b> {row['area_id']}<br>
                <b>True Rate:</b> {row['true_infection_rate']:.2f}%<br>
                <b>Predicted:</b> {row['predicted_infection_rate']:.2f}%<br>
                <b>Error:</b> {row['prediction_error']:.2f}%
                """
                
                # Add circle marker
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=size,
                    popup=folium.Popup(popup_text, max_width=300),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                    tooltip=f"Area: {row['area_id']}"
                ).add_to(m)
            
            folium_static(m, width=1000, height=600)
    
    def create_performance_analysis(self):
        """Create model performance visualizations"""
        if self.data.empty:
            st.warning("No data available for performance analysis")
            return
            
        st.markdown('<div class="section-header">📈 Model Performance Analysis</div>', unsafe_allow_html=True)
        
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
                },
                hover_data=['area_id']
            )
            
            # Add perfect prediction line
            max_rate = max(self.data['true_infection_rate'].max(), 
                          self.data['predicted_infection_rate'].max()) + 5
            fig.add_trace(go.Scatter(
                x=[0, max_rate], y=[0, max_rate],
                mode='lines',
                line=dict(dash='dash', color='gray', width=2),
                name='Perfect Prediction'
            ))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error distribution
            fig = px.histogram(
                self.data,
                x='prediction_error',
                nbins=30,
                title='Distribution of Prediction Errors',
                labels={'prediction_error': 'Prediction Error (%)'},
                color_discrete_sequence=['#ff7f0e']
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance summary
            st.subheader("Performance Summary")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("MAE", f"{self.metrics['mae']:.3f}")
            with col_b:
                st.metric("RMSE", f"{self.metrics['rmse']:.3f}")
            with col_c:
                st.metric("MSE", f"{self.metrics['mse']:.3f}")
    
    def create_feature_analysis(self):
        """Analyze feature importance and relationships"""
        if self.data.empty:
            st.warning("No data available for feature analysis")
            return
            
        st.markdown('<div class="section-header">🔍 Feature Relationship Analysis</div>', unsafe_allow_html=True)
        
        # Feature selection
        available_features = [col for col in self.data.columns if col not in 
                            ['area_id', 'data_type', 'data_quality', 'income_category', 'urban_category',
                             'prediction_abs_error', 'prediction_error']]
        
        numeric_features = []
        for feature in available_features:
            try:
                pd.to_numeric(self.data[feature])
                numeric_features.append(feature)
            except:
                pass
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis Feature", numeric_features, key="x_feature")
        
        with col2:
            y_feature = st.selectbox("Y-axis Feature", 
                                   ['true_infection_rate', 'predicted_infection_rate', 'prediction_error'], 
                                   key="y_feature")
        
        # Create scatter plot
        fig = px.scatter(
            self.data,
            x=x_feature,
            y=y_feature,
            color='data_type',
            title=f'{y_feature.replace("_", " ").title()} vs {x_feature.replace("_", " ").title()}',
            hover_data=['area_id']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        corr_features = [x_feature, y_feature, 'true_infection_rate', 'predicted_infection_rate']
        corr_features = [f for f in corr_features if f in self.data.columns]
        
        try:
            correlation = self.data[corr_features].corr()
            fig_corr = px.imshow(
                correlation,
                text_auto=True,
                aspect="auto",
                title=f'Correlation Matrix'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.warning("Could not generate correlation matrix")
    
    def create_bias_analysis(self):
        """Analyze prediction biases"""
        if self.data.empty:
            st.warning("No data available for bias analysis")
            return
            
        st.markdown('<div class="section-header">⚖️ Bias Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bias by income categories
            if 'income_category' in self.data.columns:
                bias_by_income = self.data.groupby('income_category')['prediction_error'].agg(['mean', 'std', 'count']).round(3)
                bias_by_income = bias_by_income.reset_index()
                
                fig = px.bar(
                    bias_by_income,
                    x='income_category',
                    y='mean',
                    error_y='std',
                    title='Average Prediction Error by Income Category',
                    labels={'income_category': 'Income Category', 'mean': 'Average Error (%)'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Income data not available for bias analysis")
        
        with col2:
            # Bias by urban categories
            if 'urban_category' in self.data.columns:
                bias_by_urban = self.data.groupby('urban_category')['prediction_error'].agg(['mean', 'std', 'count']).round(3)
                bias_by_urban = bias_by_urban.reset_index()
                
                fig = px.bar(
                    bias_by_urban,
                    x='urban_category',
                    y='mean',
                    error_y='std',
                    title='Average Prediction Error by Urban Category',
                    labels={'urban_category': 'Urban Category', 'mean': 'Average Error (%)'},
                    color='mean',
                    color_continuous_scale='RdBu'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Urban proximity data not available for bias analysis")
    
    def create_data_quality_analysis(self):
        """Analyze data quality impact"""
        if self.data.empty:
            return
            
        st.markdown('<div class="section-header">📋 Data Quality Impact</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data quality distribution
            quality_counts = self.data['data_type'].value_counts()
            fig = px.pie(
                values=quality_counts.values,
                names=quality_counts.index,
                title='Distribution of Data Quality Types'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance by data quality
            quality_performance = self.data.groupby('data_type').agg({
                'prediction_abs_error': 'mean',
                'true_infection_rate': 'mean',
                'area_id': 'count'
            }).round(3)
            quality_performance.columns = ['Avg Error', 'Avg True Rate', 'Area Count']
            quality_performance = quality_performance.reset_index()
            
            st.dataframe(quality_performance, use_container_width=True)
    
    def run_dashboard(self):
        """Run the complete dashboard"""
        if self.data.empty:
            st.error("Unable to load dashboard due to data issues")
            return
            
        # Header
        st.markdown('<h1 class="main-header">🦠 COVID-19 Prediction Dashboard</h1>', unsafe_allow_html=True)
        
        # Overview metrics
        self.create_overview_metrics()
        
        # Main content with tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌍 Spatial View", 
            "📊 Performance", 
            "🔍 Features", 
            "⚖️ Bias Analysis",
            "📋 Data Quality"
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
            self.create_data_quality_analysis()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>COVID-19 Integrated Prediction Model Dashboard</p>
            <p><small>Built with Streamlit | Data cleaned and processed for visualization</small></p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the dashboard"""
    # Sidebar
    st.sidebar.title("COVID-19 Dashboard")
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this Dashboard:**
    
    Visualizes COVID-19 infection rate predictions using integrated modeling that combines multiple data sources.
    
    **Features:**
    - Spatial analysis with interactive maps
    - Model performance metrics
    - Feature relationship analysis
    - Bias detection across demographics
    - Data quality impact assessment
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Stats")
    
    try:
        # Load data and run dashboard
        dashboard = COVIDDashboard('covid_integrated_predictions.csv')
        
        if not dashboard.data.empty:
            st.sidebar.metric("Total Areas", f"{len(dashboard.data):,}")
            st.sidebar.metric("Avg Infection Rate", f"{dashboard.data['true_infection_rate'].mean():.2f}%")
            st.sidebar.metric("Model R²", f"{dashboard.metrics['r2']:.3f}")
        
        dashboard.run_dashboard()
        
    except Exception as e:
        st.error(f"""
        ❌ **Application Error:**
        
        `{str(e)}`
        
        **Troubleshooting steps:**
        1. Check if `covid_integrated_predictions.csv` is in the same folder
        2. Verify the CSV file is not corrupted
        3. Ensure all required columns are present
        4. Try restarting the application
        
        If the problem persists, please check the data file format.
        """)

if __name__ == "__main__":
    main()

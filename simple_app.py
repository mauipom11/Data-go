import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Adonis Data Retrieval",
    page_icon="🧊",
    layout="wide"
)

import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from openai import OpenAI
import json
from feature import ModelTrainer

# Function to trigger a rerun
def trigger_rerun():
    st.experimental_set_query_params(rerun=True)


# Path to your CSS file
current_dir = os.path.dirname(os.path.abspath(__file__))
css_file = os.path.join(current_dir, "simple_app.css")

# Inject the CSS into the app
with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# OpenAI API setup

def get_data_analysis(query, data):
    """Get data analysis from GPT-4."""
    try:
        if client is None:
            return "Error: OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable."
            
        # Convert DataFrame info to string
        data_info = f"Columns: {', '.join(data.columns)}\n"
        data_info += f"Shape: {data.shape}\n"
        data_info += "Sample data:\n" + data.head().to_string()
        
        # Create the messages for the API
        messages = [
            {"role": "system", "content": """You are a data analysis expert. Analyze the data and provide insights.
            Always format your responses in markdown.
            For numerical analysis, include specific numbers and statistics.
            If asked to create visualizations, explain what type of chart would be best and why.
            Be concise but informative. make sure to use UK or New zealand english. and at the end generate a useful ghraph of the data"""},
            {"role": "user", "content": f"Here's the data information:\n{data_info}\n\nUser question: {query}"}
        ]
        
        # # Get response from GPT-4
        # response = client.chat.completions.create(
        #     model="gpt-4-turbo-preview",
        #     messages=messages,
        #     temperature=0.7,
        #     max_tokens=1000
        # )
               # Get response from GPT-4
        response = client.chat.completions.create(
            model="google/gemma-3-1b-it:free",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in analysis: {str(e)}"

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = None
if 'thinking' not in st.session_state:
    st.session_state.thinking = False
if 'user_input_key' not in st.session_state:
    st.session_state.user_input_key = 0
if 'show_upload' not in st.session_state:
    st.session_state.show_upload = False
if 'feature_engineering_active' not in st.session_state:
    st.session_state.feature_engineering_active = False
if 'selected_transformations' not in st.session_state:
    st.session_state.selected_transformations = {}
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# Function to reset feature engineering state
def reset_feature_engineering():
    st.session_state.feature_engineering_active = False
    st.session_state.selected_transformations = {}

# Function to activate feature engineering
def activate_feature_engineering():
    st.session_state.feature_engineering_active = True
    st.session_state.last_query = "Build Feature"

# Function to handle feature engineering button click
def handle_feature_engineering():
    activate_feature_engineering()
    st.rerun()

# Function to check feature engineering eligibility
def check_feature_eligibility(data, features):
    """Check if selected features are suitable for feature engineering."""
    checklist = []
    
    # Check data quality
    checklist.append({
        'check': 'Data Size',
        'status': '' if len(data) > 100 else '⚠️',
        'message': 'Sufficient data for feature engineering' if len(data) > 100 else 'Limited data might affect feature quality'
    })
    
    # Check for missing values
    missing_pct = data[features].isnull().mean() * 100
    for feature in features:
        status = '⚠️⚠️⚠️⚠️' if missing_pct[feature] < 5 else '⚠️⚠️⚠️⚠️⚠️⚠️'
        status = '⚠️⚠️⚠️' if missing_pct[feature] < 25 else '⚠️⚠️⚠️⚠️⚠️⚠️'
        status = '⚠️⚠️' if missing_pct[feature] < 50 else '⚠️⚠️⚠️⚠️⚠️⚠️'
        status = 'CHECKED ' if missing_pct[feature] < 90 else '⚠️⚠️⚠️⚠️⚠️⚠️'
        checklist.append({
            'check': f'Missing Values in {feature}',
            'status': status,
            'message': f'{missing_pct[feature]:.1f}% missing values'
        })
    
    # Check data types
    for feature in features:
        dtype = str(data[feature].dtype)
        checklist.append({
            'check': f'Data Type for {feature}',
            'status': '',
            'message': f'Type: {dtype}'
        })
    
    return checklist


# Sidebar with analysis tools
with st.sidebar:
    st.title("Data Explorer")

    # OpenAI API setup
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    use_openrouter = st.checkbox("Use OpenRouter API")
    if not api_key:
        st.error("OpenAI API key not found. Please enter your API key.")
        client = None
    else:
        if use_openrouter:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        else:
            client = OpenAI(api_key=api_key)
        
    # File upload section
    with st.expander("Upload Data", expanded=True):
        uploaded_file = st.file_uploader("Upload Data", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.session_state.data = data
                st.success(f"✅ Loaded {uploaded_file.name}")
                st.write(f"There's {data.shape[1]} Columns")
                st.write("and")
                st.write(f"There's {data.shape[0]:,} Rows")
                st.write("in the data file")

                # Quick column info
                numeric_cols = len(data.select_dtypes(include=['int64', 'float64']).columns)
                categorical_cols = len(data.select_dtypes(include=['object']).columns)
                st.write(f"• Numeric columns: {numeric_cols}")
                st.write(f"• Categorical columns: {categorical_cols}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.session_state.data is not None:
        # Data Info
        with st.expander("📊 Data Info", expanded=True):
            st.write(f"• Rows: {st.session_state.data.shape[0]:,}")
            st.write(f"• Columns: {st.session_state.data.shape[1]}")
            numeric_cols = len(st.session_state.data.select_dtypes(include=['int64', 'float64']).columns)
            categorical_cols = len(st.session_state.data.select_dtypes(include=['object']).columns)
            st.write(f"• Numeric columns: {numeric_cols}")
            st.write(f"• Categorical columns: {categorical_cols}")
        
        # Quick Actions
        with st.expander("⚡ Quick Actions", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Sample"):
                    st.dataframe(st.session_state.data.head(), use_container_width=True)
            with col2:
                if st.button("📊 Stats"):
                    st.dataframe(st.session_state.data.describe(), use_container_width=True)
        
        # Visualization Options
        with st.expander("📈 Visualizations", expanded=False):
            chart_type = st.selectbox(
                "Chart Type",
                ["Histogram", "Scatter Plot", "Box Plot", "Bar Chart"]
            )
            x_col = st.selectbox("Select Column", st.session_state.data.columns)
            
            if chart_type == "Scatter Plot":
                y_col = st.selectbox("vs Column", st.session_state.data.columns)
            
            if st.button("📊 Generate Chart"):
                try:
                    if chart_type == "Histogram":
                        fig = px.histogram(st.session_state.data, x=x_col, title=f"Histogram of {x_col}")
                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(st.session_state.data, x=x_col, y=y_col, 
                                       title=f"Scatter Plot: {x_col} vs {y_col}")
                    elif chart_type == "Box Plot":
                        fig = px.box(st.session_state.data, y=x_col, title=f"Box Plot of {x_col}")
                    else:  # Bar Chart
                        if st.session_state.data[x_col].dtype == 'object':
                            counts = st.session_state.data[x_col].value_counts()
                            fig = px.bar(x=counts.index, y=counts.values, 
                                       title=f"Bar Chart of {x_col}")
                        else:
                            fig = px.bar(st.session_state.data, x=x_col, 
                                       title=f"Bar Chart of {x_col}")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating chart: {str(e)}")

# Main chat area title (always visible)
st.title("✨ Data Explorer Chat")

# Welcome message and initial buttons (only shown when no data)
if st.session_state.data is None or st.session_state.data.empty:
    st.markdown("""
        <div style="text-align: center; padding: 2rem; animation: fadeIn 1s ease;">
            <h1 style="margin-bottom: 1rem;">👋 Welcome to Data Explorer!</h1>
            <p style="color: var(--text-color); font-size: 1.2em; margin: 1rem 0;">
                I'm here to help you analyze and understand your data. Please upload a file to get started.
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Show action buttons when data is loaded
    st.markdown("### Choose an Action:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Find Info", key="main_info", use_container_width=True):
            st.session_state.last_query = "Find Info"
            reset_feature_engineering()
            st.rerun()

    with col2:
        if st.button("⚡ Build Feature", key="main_feature", use_container_width=True):
            activate_feature_engineering()

# Display chat history
for msg in st.session_state.chat_history:
    if msg['type'] == 'user':
        st.markdown(f"""<div class="user-message">💭 {msg['text']}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="assistant-message">✨ {msg['text']}</div>""", unsafe_allow_html=True)

# Create the text input with a key that changes when we want to clear it
user_input = None
if st.session_state.data is not None:
    user_input = st.text_input("Ask a question", 
                            placeholder="Ask anything about your data... (Press Enter to send)",
                            key=f"chat_input_{st.session_state.user_input_key}",
                            on_change=lambda: None)

# Handle Enter key press or button clicks
if (user_input and user_input != st.session_state.last_query) or st.session_state.last_query in ["Find Info", "Find Value", "Build Feature"]:
    current_query = user_input if user_input else st.session_state.last_query
    if current_query != st.session_state.last_query:
        st.session_state.last_query = current_query
        st.session_state.chat_history.append({'type': 'user', 'text': current_query})
    st.session_state.thinking = True
    # Increment the key to clear the input on next render
    st.session_state.user_input_key += 1
    
    try:
        if st.session_state.thinking:
            st.markdown('<div class="thinking">✨ Analyzing your data...</div>', unsafe_allow_html=True)
        
        # First show the text response
        if st.session_state.data is None or st.session_state.data.empty:
            response = "I notice you haven't uploaded any data yet. Please upload a file to get started."
            st.markdown(f"""<div class="assistant-message">✨ {response}</div>""", unsafe_allow_html=True)
        else:
            # Handle different button actions
            if 'Find Info' in current_query:
                response = get_data_analysis("Give me a summary of the data and its main characteristics", st.session_state.data)
                st.session_state.feature_engineering_active = False
            elif 'Find Value' in current_query:
                response = get_data_analysis("What are the key insights and potential value in this data?", st.session_state.data)
                st.session_state.feature_engineering_active = False
            elif 'Build Feature' in current_query:
                response = "Let's create some new features! I'll help you through the process."
                activate_feature_engineering()
            else:
                response = get_data_analysis(current_query, st.session_state.data)
                st.session_state.feature_engineering_active = False
            
            # Display the response
            st.markdown(f"""<div class="assistant-message">✨ {response}</div>""", unsafe_allow_html=True)
            
            # Show feature engineering interface if active
            if st.session_state.feature_engineering_active and st.session_state.data is not None:
                # Feature Engineering UI
                st.markdown("### 🛠️ Feature Engineering")
                
                # Feature selection
                st.markdown("#### Select Features")
                available_features = st.session_state.data.columns.tolist()
                selected_features = st.multiselect("Choose features to work with:", available_features)
                
                # Filter out non-numeric features for conversion check
                numeric_features = [feature for feature in selected_features if np.issubdtype(st.session_state.data[feature].dtype, np.number)]
                
                # Check for features containing '$' and convert them
                dollar_features = [feature for feature in numeric_features if st.session_state.data[feature].astype(str).str.contains(r'\$').any()]
                
                if dollar_features:
                    st.markdown("#### Features containing '$'")
                    for feature in dollar_features:
                        st.write(f"Feature: {feature}")
                        if st.button(f"Convert {feature} from $ to numeric"):
                            st.session_state.data[feature] = st.session_state.data[feature].replace(r'[\$,]', '', regex=True).astype(float)
                            st.success(f"Converted {feature} to numeric values.")
                            trigger_rerun()
                
                if selected_features:
                    sample_size = st.slider("Sample size to display:", min_value=5, max_value=len(st.session_state.data), value=5)
                    # Feature value search
                    st.markdown("#### Feature Value Search")
                    search_feature = st.selectbox("Select Feature to Search", selected_features)
                    search_value = st.text_input(f"Enter value to search in {search_feature}")
                    if search_value:
                        try:                            
                            search_results = st.session_state.data[st.session_state.data[search_feature].astype(str).str.contains(search_value, case=False, na=False)]
                            st.markdown(f"#### Search Results for '{search_value}' in {search_feature}")
                            if not search_results.empty:
                                st.dataframe(search_results[selected_features].head(sample_size))  # Use sample_size from the slider
                            else:
                                st.warning(f"No results found for '{search_value}' in {search_feature}")
                        except Exception as e:
                            st.error(f"Error searching for value: {str(e)}")
                    # Display selected features
                    st.markdown("#### Selected Features Preview")
                    st.dataframe(st.session_state.data[selected_features].head(sample_size))  # Use sample_size from the slider
                
                    # Run eligibility checks
                    checklist = check_feature_eligibility(st.session_state.data, selected_features)
                    
                    # Display checklist
                    st.markdown("#### Feature Engineering Checklist")
                    for check in checklist:
                        st.markdown(f"{check['status']} **{check['check']}**: {check['message']}")
                    
                    # Feature engineering suggestions with clickable buttons
                    st.markdown("#### Suggested Transformations")
                    for feature in selected_features:
                        dtype = st.session_state.data[feature].dtype
                        st.markdown(f"**{feature}** (Type: {dtype})")
                        
                        # Create columns for transformation buttons
                        if np.issubdtype(dtype, np.number):
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("📊 Standardization", key=f"std_{feature}"):
                                    st.session_state.selected_transformations[feature] = "standardization"
                                if st.button("📈 Log Transform", key=f"log_{feature}"):
                                    st.session_state.selected_transformations[feature] = "log"
                            with col2:
                                if st.button("🔄 Min-Max Scaling", key=f"minmax_{feature}"):
                                    st.session_state.selected_transformations[feature] = "minmax"
                                if st.button("📦 Binning", key=f"bin_{feature}"):
                                    st.session_state.selected_transformations[feature] = "binning"
                        
                        elif dtype == 'object':
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🎯 One-Hot Encoding", key=f"onehot_{feature}"):
                                    st.session_state.selected_transformations[feature] = "onehot"
                                if st.button("🏷️ Label Encoding", key=f"label_{feature}"):
                                    st.session_state.selected_transformations[feature] = "label"
                            with col2:
                                if st.button("📊 Frequency Encoding", key=f"freq_{feature}"):
                                    st.session_state.selected_transformations[feature] = "frequency"
                                if st.button("🎯 Target Encoding", key=f"target_{feature}"):
                                    st.session_state.selected_transformations[feature] = "target"
                        
                        # Show selected transformation
                        if feature in st.session_state.selected_transformations:
                            st.info(f"Selected transformation: {st.session_state.selected_transformations[feature]}")
                    
                    # Show Output button if any transformation is selected
                    if st.session_state.selected_transformations:
                        st.markdown("#### Generate Transformed Features")
                        
                        # Add target column selection
                        target_column = st.selectbox(
                            "Select Target Column",
                            [col for col in st.session_state.data.columns if col not in selected_features],
                            help="Select the column you want to predict"
                        )
                        
                        # Add problem type selection
                        problem_type = st.selectbox(
                            "Select Problem Type",
                            ["regression", "classification"],
                            help="Choose regression for continuous targets, classification for categorical targets"
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔄 Generate Output", key="generate_output"):
                                st.markdown("### Preview of Transformed Features")
                                
                                # Initialize model trainer
                                st.session_state.model_trainer = ModelTrainer()
                                
                                # Train model and get results
                                with st.spinner("Training model and generating Data Card..."):
                                    st.session_state.model_results = st.session_state.model_trainer.train_model(
                                        st.session_state.data,
                                        target_column,
                                        st.session_state.selected_transformations,
                                        problem_type
                                    )
                                
                                # Display results
                                st.success("Model training complete!")
                                
                                # Show metrics
                                st.markdown("#### Model Performance")
                                for metric, value in st.session_state.model_results["metrics"].items():
                                    st.metric(metric, f"{value:.4f}")
                                
                                # Show feature importance
                                st.markdown("#### Feature Importance")
                                importance_df = pd.DataFrame(
                                    st.session_state.model_results["feature_importance"].items(),
                                    columns=["Feature", "Importance"]
                                ).sort_values("Importance", ascending=False)
                                st.dataframe(importance_df)
                        
                        with col2:
                            if st.session_state.model_results is not None:
                                if st.button("📥 Download Data Card"):
                                    data_card = st.session_state.model_trainer.generate_data_card()
                                    st.download_button(
                                        "📄 Download Data Card as Markdown",
                                        data_card,
                                        file_name="data_card.md",
                                        mime="text/markdown"
                                    )
                        
                        if st.session_state.model_results is not None:
                            # Show detailed logs
                            st.markdown("#### Training Log")
                            for log in st.session_state.model_results["log_entries"]:
                                st.markdown(f"""
                                **{log['step']}** - {log['timestamp']}  
                                {log['description']}  
                                {f"Details: {log['details']}" if log['details'] else ""}
                                """)
                            
                            # Model Suggestions and Recommendations
                            st.markdown("### 🤖 Model Insights")
                            
                            # Get feature types after transformation
                            numeric_features = [f for f in selected_features 
                                             if np.issubdtype(st.session_state.data[f].dtype, np.number)]
                            categorical_features = [f for f in selected_features 
                                                 if st.session_state.data[f].dtype == 'object']
                            
                            # Provide model suggestions based on data characteristics
                            st.markdown("#### Recommended Models")
                            if len(numeric_features) > 0 and len(categorical_features) == 0:
                                st.markdown("""
                                - **Linear Models**: Good for interpretable results
                                    - Linear Regression (for continuous targets)
                                    - Logistic Regression (for binary classification)
                                - **Tree-Based Models**: Can capture non-linear relationships
                                    - Random Forest
                                    - XGBoost
                                - **Neural Networks**: For complex patterns in large datasets
                                """)
                            elif len(categorical_features) > 0:
                                st.markdown("""
                                - **Tree-Based Models**: Excellent for mixed data types
                                    - Random Forest
                                    - XGBoost
                                    - LightGBM
                                - **Neural Networks**: With appropriate embeddings for categorical features
                                """)
                            
                            # Usage Recommendations
                            st.markdown("#### How to Use Transformed Features")
                            st.markdown("""
                            1. **Data Split**: 
                               - Split data into training (70%), validation (15%), and test (15%) sets
                               - Apply the same transformations to all splits
                            
                            2. **Model Training**:
                               - Start with simpler models (e.g., Random Forest)
                               - Use cross-validation to assess model stability
                               - Monitor for overfitting
                            
                            3. **Feature Importance**:
                               - Analyze which transformed features contribute most
                               - Consider removing less important features
                            
                            4. **Model Evaluation**:
                               - Use appropriate metrics (e.g., RMSE for regression, AUC-ROC for classification)
                               - Compare performance with and without transformed features
                            """)
                    
                    # User input for custom ideas
                    st.markdown("#### Share Your Ideas")
                    user_suggestion = st.text_area(
                        "What transformations do you think would be useful? Why?",
                        help="Describe your ideas for feature engineering based on your domain knowledge."
                    )
                    
                    if user_suggestion:
                        # Get AI feedback on user suggestion
                        feedback = get_data_analysis(
                            f"Evaluate this feature engineering suggestion for {', '.join(selected_features)}: {user_suggestion}",
                            st.session_state.data[selected_features]
                        )
                        st.markdown("#### AI Feedback on Your Suggestion")
                        st.markdown(feedback)
            
            # Show action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔍 Find Info", key="btn_info", use_container_width=True):
                    st.session_state.last_query = "Find Info"
                    reset_feature_engineering()
                    st.rerun()
            with col2:
                if st.button("📊 Find Value", key="btn_value", use_container_width=True):
                    st.session_state.last_query = "Find Value"
                    reset_feature_engineering()
                    st.rerun()
            with col3:
                if st.button("⚡ Build Feature", key="btn_feature", use_container_width=True):
                    handle_feature_engineering()
        
        # Add response to chat history AFTER showing everything
        st.session_state.thinking = False
        if current_query != st.session_state.last_query:
            st.session_state.chat_history.append({'type': 'assistant', 'text': response})
        
    except Exception as e:
        st.session_state.thinking = False
        response = f"I encountered an error: {str(e)}\nLet's try something else:"
        st.markdown(f"""<div class="assistant-message">✨ {response}</div>""", unsafe_allow_html=True)
        st.session_state.chat_history.append({'type': 'assistant', 'text': response})
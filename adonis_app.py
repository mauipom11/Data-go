import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Adonis Data Retrieval",
    page_icon="🧊",
    layout="wide"
)

# Path to your CSS file
current_dir = os.path.dirname(os.path.abspath(__file__))
css_file = os.path.join(current_dir, "simple_app.css")

# Inject the CSS into the app
with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

# Sidebar with analysis tools
with st.sidebar:
    st.title("Data Explorer")
    
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

# Welcome message at the top (only shown when no data)
if st.session_state.data is None or st.session_state.data.empty:
    st.markdown("""
        <div style="text-align: center; padding: 2rem; animation: fadeIn 1s ease;">
            <h1 style="margin-bottom: 1rem;">👋 Welcome to Data Explorer!</h1>
            <p style="color: var(--text-color); font-size: 1.2em; margin: 1rem 0;">
                I'm here to help you analyze and understand your data. You can:
            </p>
        </div>
        <div class="suggestion-container" style="justify-content: center; margin-bottom: 2rem;">
            <button class="suggestion-button">Learn features</button>
            <button class="suggestion-button">Get started</button>
        </div>
    """, unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.chat_history:
    if msg['type'] == 'user':
        st.markdown(f"""<div class="user-message">💭 {msg['text']}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="assistant-message">✨ {msg['text']}</div>""", unsafe_allow_html=True)

# Create the text input with a key that changes when we want to clear it
user_input = st.text_input("", 
                          placeholder="Ask anything about your data... (Press Enter to send)",
                          key=f"chat_input_{st.session_state.user_input_key}",
                          on_change=lambda: None)

# Handle Enter key press
if user_input and user_input != st.session_state.last_query:
    st.session_state.last_query = user_input
    st.session_state.chat_history.append({'type': 'user', 'text': user_input})
    st.session_state.thinking = True
    # Increment the key to clear the input on next render
    st.session_state.user_input_key += 1
    
    try:
        if st.session_state.thinking:
            st.markdown('<div class="thinking">✨ Analyzing your data...</div>', unsafe_allow_html=True)
        
        # First show the text response
        if st.session_state.data is None or st.session_state.data.empty:
            response = "I notice you haven't uploaded any data yet. You can:"
            st.markdown(f"""<div class="assistant-message">✨ {response}</div>""", unsafe_allow_html=True)
            # Then show the buttons
            st.markdown("""
                <div class="suggestion-container">
                    <button class="suggestion-button">📤 Upload CSV</button>
                    <button class="suggestion-button">📊 Upload Excel</button>
                    <button class="suggestion-button">❓ Learn More</button>
                </div>
            """, unsafe_allow_html=True)
        elif 'hi' in user_input.lower() or 'hello' in user_input.lower():
            response = "Hello! How can I help you analyze your data today?"
            st.markdown(f"""<div class="assistant-message">✨ {response}</div>""", unsafe_allow_html=True)
            st.markdown("""
                <div class="suggestion-container">
                    <button class="suggestion-button">📊 Show Summary</button>
                    <button class="suggestion-button">🔍 Find Insights</button>
                    <button class="suggestion-button">📈 Create Charts</button>
                </div>
            """, unsafe_allow_html=True)
        elif 'distribution' in user_input.lower():
            response = "Here are the distributions of your numeric columns. Would you like to:"
            st.markdown(f"""<div class="assistant-message">✨ {response}</div>""", unsafe_allow_html=True)
            
            # Show the plots
            numeric_cols = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols[:3]:
                fig = px.histogram(st.session_state.data, x=col, 
                                 title=f"Distribution of {col}",
                                 template="seaborn")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Then show the buttons
            st.markdown("""
                <div class="suggestion-container">
                    <button class="suggestion-button">📊 More Columns</button>
                    <button class="suggestion-button">📈 Add Trendlines</button>
                    <button class="suggestion-button">🔄 Compare</button>
                </div>
            """, unsafe_allow_html=True)
        else:
            response = "Here are some things I can help you with:"
            st.markdown(f"""<div class="assistant-message">✨ {response}</div>""", unsafe_allow_html=True)
            st.markdown("""
                <div class="suggestion-container">
                    <button class="suggestion-button">📊 Analyze Data</button>
                    <button class="suggestion-button">📈 Create Charts</button>
                    <button class="suggestion-button">🔍 Find Patterns</button>
                </div>
            """, unsafe_allow_html=True)
        
        # Add response to chat history AFTER showing everything
        st.session_state.thinking = False
        st.session_state.chat_history.append({'type': 'assistant', 'text': response})
        
        if 'distribution' in user_input.lower():
            col1, col2, col3 = st.columns(3)
            if col1.button("📊 More Columns"):
                st.session_state.last_query = "Show more column distributions"
                st.rerun()
            if col2.button("📈 Add Trendlines"):
                st.session_state.last_query = "Add trend lines to distributions"
                st.rerun()
            if col3.button("🔄 Compare Columns"):
                st.session_state.last_query = "Compare column distributions"
                st.rerun()
        
        elif 'relationship' in user_input.lower():
            col1, col2, col3 = st.columns(3)
            if col1.button("🔍 Strongest Correlation"):
                st.session_state.last_query = "Show strongest correlations"
                st.rerun()
            if col2.button("📊 Scatter Plots"):
                st.session_state.last_query = "Create scatter plots"
                st.rerun()
            if col3.button("📈 Trend Analysis"):
                st.session_state.last_query = "Analyze trends"
                st.rerun()
        
    except Exception as e:
        st.session_state.thinking = False
        response = f"I encountered an error: {str(e)}\nLet's try something else:"
        st.markdown("""
            <div class="suggestion-container">
                <button class="suggestion-button">Show basic stats</button>
                <button class="suggestion-button">Visualize data</button>
                <button class="suggestion-button">Check data quality</button>
            </div>
        """, unsafe_allow_html=True)
        st.session_state.chat_history.append({'type': 'assistant', 'text': response})
        st.rerun()
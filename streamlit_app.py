import streamlit as st
import requests
import pandas as pd
import json
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuration
GROQ_API_KEY = "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"
DATA_GOV_API_KEY = "579b464db66ec23bdd0000010edcdeee7a1f4a397903c72ca4686ae5"
BASE_URL = "ht" \
"tps://api.data.gov.in/resource"

# Dataset configurations
DATASETS = {
    "crop_production": {
        "id": "35be999b-0208-4354-b557-f6ca9a5355de",
        "name": "District-wise, Season-wise Crop Production Statistics (1997+)",
        "fields": ["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production"],
        "description": "Comprehensive crop production data across Indian districts since 1997"
    },
    "vegetable_crops": {
        "id": "d6e5315d-d4a7-4f1f-ab23-c2adcac3e1e7",
        "name": "District Wise Area Production Yield Value for Vegetable Crops 2021",
        "fields": ["State", "District", "Crop", "Area", "Production", "Yield", "Value"],
        "description": "Detailed vegetable crop statistics for 2021"
    }
}

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    client = Groq(api_key=GROQ_API_KEY)
    return client

# Fetch data from data.gov.in with retry logic
@st.cache_data(ttl=3600)
def fetch_data(dataset_id, filters=None, limit=1000, offset=0):
    """Fetch data from data.gov.in API with retry logic"""
    url = f"{BASE_URL}/{dataset_id}"
    params = {
        "api-key": DATA_GOV_API_KEY,
        "format": "json",
        "limit": limit,
        "offset": offset
    }
    
    if filters:
        params["filters"] = json.dumps(filters)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "records" in data:
                df = pd.DataFrame(data["records"])
                # Clean column names
                df.columns = df.columns.str.strip()
                return df
            return pd.DataFrame()
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            st.error("Request timed out. Please try again.")
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return pd.DataFrame()

# Analyze query using LLM
def analyze_query(question, available_datasets):
    """Use LLM to understand the query and determine data requirements"""
    client = get_groq_client()
    
    dataset_info = "\n".join([
        f"- {key}: {ds['name']}\n  Fields: {', '.join(ds['fields'])}\n  {ds['description']}"
        for key, ds in available_datasets.items()
    ])
    
    system_prompt = f"""You are an intelligent data analyst for India's agricultural data portal.

Available datasets:
{dataset_info}

Analyze the user's question and respond with ONLY a valid JSON object (no markdown, no explanations) containing:
1. "datasets_needed": Array of dataset IDs required (use keys: "crop_production" or "vegetable_crops")
2. "filters": Object with filters for each dataset (use exact field names from the dataset)
3. "analysis_type": Type of analysis (comparison, trend, ranking, correlation, summary)
4. "entities": Object with extracted entities
   - "states": Array of state names (use proper capitalization)
   - "crops": Array of crop names
   - "districts": Array of district names
   - "years": Array or range of years
   - "seasons": Array of seasons if applicable
5. "query_plan": Brief step-by-step plan to answer the question

Important: 
- Use exact field names: "State_Name" for crop_production, "State" for vegetable_crops
- Match state names exactly as they appear in government data (e.g., "Punjab", "Uttar Pradesh")
- For year ranges, extract start and end years
- Return ONLY the JSON, no additional text"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this question: {question}"}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Remove any leading/trailing text
        content = content[content.find('{'):content.rfind('}')+1]
        
        return json.loads(content)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing LLM response: {str(e)}")
        st.code(content)
        return None
    except Exception as e:
        st.error(f"Error analyzing query: {str(e)}")
        return None

# Generate answer using LLM
def generate_answer(question, data_summary, query_analysis, dataframes):
    """Generate natural language answer with citations"""
    client = get_groq_client()
    
    # Create detailed data summary
    detailed_summary = data_summary + "\n\n**Sample Data:**\n"
    for ds_name, df in dataframes.items():
        if not df.empty:
            detailed_summary += f"\n{DATASETS[ds_name]['name']}:\n"
            detailed_summary += df.head(5).to_string() + "\n"
    
    system_prompt = """You are an expert agricultural data analyst providing insights based on government data.

CRITICAL REQUIREMENTS:
1. Answer ONLY based on the data provided - do not make up information
2. Cite EVERY specific data point with [Source: Dataset_Name]
3. Use exact numbers from the data
4. If data is insufficient, clearly state limitations
5. Provide insights, comparisons, and trends
6. Format numbers with commas for readability (e.g., 1,234,567)
7. Include units (tonnes, hectares, etc.) where applicable
8. Be precise and accurate

Answer format:
- Start with a direct answer to the question
- Provide supporting data with citations
- Add analysis or insights
- Mention any data limitations or caveats"""

    user_prompt = f"""Question: {question}

Query Analysis:
{json.dumps(query_analysis, indent=2)}

Available Data:
{detailed_summary}

Provide a comprehensive, accurate answer with proper citations."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

# Page configuration
st.set_page_config(
    page_title="Project Samarth - Agricultural Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .dataset-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E7D32;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .dataset-card:hover {
        transform: translateX(5px);
    }
    .metric-card {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .citation {
        background: #E8F5E9;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #4CAF50;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46,125,50,0.3);
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-success {
        background: #C8E6C9;
        color: #2E7D32;
    }
    .status-processing {
        background: #FFF9C4;
        color: #F57F17;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌾 Project Samarth</h1>
    <p style="font-size: 1.2rem; margin: 0; font-weight: 500;">Intelligent Q&A System for India's Agricultural Data</p>
    <p style="font-size: 0.95rem; opacity: 0.95; margin-top: 0.5rem;">
        Powered by data.gov.in | Ministry of Agriculture & Farmers Welfare | AI-Driven Insights
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 System Information")
    
    st.markdown("#### Available Datasets")
    for key, dataset in DATASETS.items():
        st.markdown(f"""
        <div class="dataset-card">
            <strong>🗂️ {dataset['name']}</strong><br>
            <small style="opacity: 0.7;">{dataset['description']}</small><br>
            <span class="status-badge status-success">● Active</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Sample Questions")
    
    sample_questions = [
        "What are the top 5 crops by production in Maharashtra in 2020?",
        "Compare wheat production between Punjab and Haryana from 2015 to 2020",
        "Which district in Uttar Pradesh has the highest rice production?",
        "Show me vegetable production data for Karnataka in 2021",
        "List all crops grown in Kharif season in Bihar",
        "What is the total area under cultivation for cotton in Gujarat?"
    ]
    
    for i, q in enumerate(sample_questions, 1):
        if st.button(f"📝 {q[:30]}...", key=f"sample_{i}", use_container_width=True):
            st.session_state.question = q
            st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    **Ask complex questions!**
    - Compare across states
    - Analyze trends over years
    - Find top/bottom performers
    - Explore seasonal patterns
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Datasets", "2", delta="Active")
    with col2:
        st.metric("API", "Live", delta="Connected")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'question' not in st.session_state:
    st.session_state.question = ""

# Main content area
tab1, tab2, tab3 = st.tabs(["🔍 Query", "📊 Data Explorer", "📜 History"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Ask Your Question")
        
        question = st.text_area(
            "Enter your question about Indian agriculture:",
            value=st.session_state.question,
            height=120,
            placeholder="e.g., Compare rice production in West Bengal and Odisha for the last 5 years",
            help="Ask questions about crop production, yields, areas, states, districts, and more!"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        with col_btn1:
            submit_btn = st.button("🚀 Analyze Query", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        with col_btn3:
            if st.button("🔄 Reset", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")

    with col2:
        st.markdown("### 📈 Quick Stats")
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <h2 style="margin: 0;">{len(DATASETS)}</h2>
            <p style="margin: 0; opacity: 0.9;">Active Datasets</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0;">{len(st.session_state.history)}</h2>
            <p style="margin: 0; opacity: 0.9;">Queries Processed</p>
        </div>
        """, unsafe_allow_html=True)
    
    if clear_btn:
        st.session_state.question = ""
        st.rerun()

    # Process query
    if submit_btn and question:
        st.markdown("---")
        
        # Step 1: Analyze query
        with st.spinner("🧠 Analyzing your question..."):
            st.markdown("### 🧠 Query Analysis")
            analysis = analyze_query(question, DATASETS)
        
        if analysis:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**📋 Detected Entities:**")
                entities = analysis.get("entities", {})
                if entities.get("states"):
                    st.write(f"🗺️ **States:** {', '.join(entities['states'])}")
                if entities.get("crops"):
                    st.write(f"🌾 **Crops:** {', '.join(entities['crops'])}")
                if entities.get("years"):
                    st.write(f"📅 **Years:** {entities['years']}")
                if entities.get("districts"):
                    st.write(f"📍 **Districts:** {', '.join(entities['districts'])}")
            
            with col2:
                st.markdown("**🎯 Analysis Type:**")
                st.info(f"**{analysis.get('analysis_type', 'Unknown').title()}**")
                st.markdown("**📊 Datasets Required:**")
                for ds in analysis.get("datasets_needed", []):
                    if ds in DATASETS:
                        st.write(f"✓ {DATASETS[ds]['name']}")
            
            with st.expander("🔍 View Detailed Analysis", expanded=False):
                st.json(analysis)
            
            # Step 2: Fetch required data
            st.markdown("### 📥 Fetching Data from data.gov.in")
            all_data = {}
            data_summary = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            datasets_to_fetch = analysis.get("datasets_needed", [])
            
            for i, dataset_key in enumerate(datasets_to_fetch):
                if dataset_key in DATASETS:
                    dataset_info = DATASETS[dataset_key]
                    status_text.markdown(f'<span class="status-badge status-processing">⏳ Fetching: {dataset_info["name"]}</span>', unsafe_allow_html=True)
                    
                    # Get filters for this dataset
                    filters = analysis.get("filters", {}).get(dataset_key, {})
                    
                    # Fetch data
                    df = fetch_data(dataset_info["id"], filters=filters, limit=10000)
                    
                    if not df.empty:
                        all_data[dataset_key] = df
                        
                        # Create summary
                        summary = f"\n**{dataset_info['name']}**:\n"
                        summary += f"- Records fetched: {len(df):,}\n"
                        summary += f"- Columns: {', '.join(df.columns.tolist()[:10])}\n"
                        
                        # Add statistical summary
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        for col in ['Production', 'Area', 'Yield']:
                            if col in df.columns:
                                total = df[col].sum()
                                summary += f"- Total {col}: {total:,.2f}\n"
                        
                        data_summary.append(summary)
                        
                        # Display preview
                        with st.expander(f"📊 Preview: {dataset_info['name']} ({len(df):,} records)", expanded=False):
                            st.dataframe(df.head(20), use_container_width=True, height=300)
                            
                            # Display statistics
                            if not numeric_cols.empty:
                                st.markdown("**Statistical Summary:**")
                                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    else:
                        st.warning(f"⚠️ No data returned for {dataset_info['name']}")
                
                progress_bar.progress((i + 1) / len(datasets_to_fetch))
            
            status_text.markdown('<span class="status-badge status-success">✓ Data Fetching Complete</span>', unsafe_allow_html=True)
            
            # Step 3: Generate answer
            if all_data:
                st.markdown("### 🎯 Intelligent Answer")
                
                with st.spinner("🤖 Generating comprehensive answer with citations..."):
                    combined_summary = "\n".join(data_summary)
                    answer = generate_answer(question, combined_summary, analysis, all_data)
                
                if answer:
                    # Display answer in a nice container
                    st.markdown("""
                    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #2E7D32;">
                    """, unsafe_allow_html=True)
                    st.markdown(answer)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": question,
                        "answer": answer,
                        "datasets_used": len(all_data),
                        "records_processed": sum(len(df) for df in all_data.values())
                    })
                    
                    # Visualizations
                    st.markdown("### 📊 Data Visualizations")
                    
                    viz_tabs = st.tabs([f"📈 {DATASETS[key]['name'][:30]}..." for key in all_data.keys()])
                    
                    for idx, (dataset_key, df) in enumerate(all_data.items()):
                        with viz_tabs[idx]:
                            if not df.empty:
                                # Auto-generate appropriate charts
                                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                                
                                if 'Production' in df.columns and 'Crop' in df.columns:
                                    # Top crops by production
                                    top_crops = df.groupby('Crop')['Production'].sum().nlargest(15).reset_index()
                                    fig = px.bar(
                                        top_crops,
                                        x='Crop',
                                        y='Production',
                                        title=f'Top 15 Crops by Production',
                                        labels={'Production': 'Production (Tonnes)', 'Crop': 'Crop Type'},
                                        color='Production',
                                        color_continuous_scale='Greens',
                                        text='Production'
                                    )
                                    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                                    fig.update_layout(xaxis_tickangle=-45, height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                if 'Production' in df.columns and 'State_Name' in df.columns:
                                    # State-wise production
                                    state_prod = df.groupby('State_Name')['Production'].sum().nlargest(15).reset_index()
                                    fig = px.bar(
                                        state_prod,
                                        y='State_Name',
                                        x='Production',
                                        orientation='h',
                                        title='Top 15 States by Production',
                                        labels={'Production': 'Production (Tonnes)', 'State_Name': 'State'},
                                        color='Production',
                                        color_continuous_scale='Blues',
                                        text='Production'
                                    )
                                    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                                    fig.update_layout(height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                if 'Crop_Year' in df.columns and 'Production' in df.columns:
                                    # Yearly trend
                                    yearly = df.groupby('Crop_Year')['Production'].sum().reset_index()
                                    yearly = yearly.sort_values('Crop_Year')
                                    fig = px.line(
                                        yearly,
                                        x='Crop_Year',
                                        y='Production',
                                        title='Production Trend Over Years',
                                        labels={'Production': 'Production (Tonnes)', 'Crop_Year': 'Year'},
                                        markers=True
                                    )
                                    fig.update_traces(line_color='#2E7D32', line_width=3, marker_size=8)
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Show data summary
                                st.markdown("#### 📋 Data Summary")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Records", f"{len(df):,}")
                                with col2:
                                    if 'Production' in df.columns:
                                        st.metric("Total Production", f"{df['Production'].sum():,.0f}")
                                with col3:
                                    if 'Area' in df.columns:
                                        st.metric("Total Area", f"{df['Area'].sum():,.0f}")
                    
                    # Download section
                    st.markdown("### 💾 Download Data")
                    dl_cols = st.columns(len(all_data))
                    for idx, (dataset_key, df) in enumerate(all_data.items()):
                        with dl_cols[idx]:
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=f"📥 {DATASETS[dataset_key]['name'][:20]}...",
                                data=csv,
                                file_name=f"{dataset_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key=f"download_{dataset_key}",
                                use_container_width=True
                            )
            else:
                st.error("❌ No data could be retrieved. This might be because:")
                st.write("- The filters were too restrictive")
                st.write("- The data doesn't exist for the specified criteria")
                st.write("- There was an API error")
                st.info("💡 Try broadening your query or using different parameters")

with tab2:
    st.markdown("### 📊 Data Explorer")
    st.info("Select a dataset to explore its structure and sample data")
    
    selected_dataset = st.selectbox(
        "Choose a dataset:",
        options=list(DATASETS.keys()),
        format_func=lambda x: DATASETS[x]['name']
    )
    
    if selected_dataset:
        ds_info = DATASETS[selected_dataset]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**{ds_info['name']}**")
            st.write(ds_info['description'])
        
        with col2:
            st.markdown("**Available Fields:**")
            for field in ds_info['fields']:
                st.write(f"• {field}")
        
        if st.button("🔍 Load Sample Data", key="explore_btn"):
            with st.spinner("Loading data..."):
                df = fetch_data(ds_info['id'], limit=100)
                
                if not df.empty:
                    st.success(f"✓ Loaded {len(df)} sample records")
                    
                    st.markdown("#### Sample Records")
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    st.markdown("#### Data Statistics")
                    st.dataframe(df.describe(), use_container_width=True)
                else:
                    st.error("Could not load data")

with tab3:
    st.markdown("### 📜 Query History")
    
    if st.session_state.history:
        st.success(f"You have {len(st.session_state.history)} queries in history")
        
        for i, item in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Query #{len(st.session_state.history) - i + 1}: {item['question'][:80]}...", expanded=(i == 1)):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏰ Timestamp", item['timestamp'])
                with col2:
                    st.metric("📊 Datasets Used", item['datasets_used'])
                with col3:
                    st.metric("📝 Records Processed", f"{item['records_processed']:,}")
                
                st.markdown("**Question:**")
                st.info(item['question'])
                
                st.markdown("**Answer:**")
                st.markdown(item['answer'])
    else:
        st.info("No queries yet. Start by asking a question in the Query tab!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
    <p style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">🌾 Project Samarth</p>
    <p style="margin: 0.25rem 0;">Intelligent Agricultural Data Analytics for Policy Makers</p>
    <p style="font-size: 0.9rem; margin: 0.25rem 0;">
        Data Source: <a href="https://data.gov.in" target="_blank" style="color: #2E7D32;">data.gov.in</a> | 
        AI Powered by <strong>Groq (Llama 3.3 70B)</strong>
    </p>
    <p style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.8;">
        Built with ❤️ for better agricultural insights and data-driven policy decisions
    </p>
</div>
""", unsafe_allow_html=True)

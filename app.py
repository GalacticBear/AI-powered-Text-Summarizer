import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import pandas as pd
import time
import json
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📝 AI-Powered Text Summarizer")
st.markdown("""
This application uses state-of-the-art transformer models to generate high-quality summaries from long texts.
Choose from multiple models and compare their performance with ROUGE score evaluation.
""")

@st.cache_resource
def load_model(model_name):
    """Load and cache the summarization model"""
    try:
        if model_name == "facebook/bart-large-cnn":
            summarizer = pipeline("summarization", model=model_name, 
                                tokenizer=model_name, device=0 if torch.cuda.is_available() else -1)
        elif model_name.startswith("t5"):
            summarizer = pipeline("text2text-generation", model=model_name,
                                tokenizer=model_name, device=0 if torch.cuda.is_available() else -1)
        else:
            summarizer = pipeline("summarization", model=model_name,
                                device=0 if torch.cuda.is_available() else -1)
        return summarizer
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None

def calculate_rouge_scores(generated_summary, reference_summary):
    """Calculate ROUGE scores between generated and reference summaries"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(generated_summary, reference_summary)
    
    return {
        'ROUGE-1': {
            'Precision': scores['rouge1'].precision,
            'Recall': scores['rouge1'].recall,
            'F1-Score': scores['rouge1'].fmeasure
        },
        'ROUGE-2': {
            'Precision': scores['rouge2'].precision,
            'Recall': scores['rouge2'].recall,
            'F1-Score': scores['rouge2'].fmeasure
        },
        'ROUGE-L': {
            'Precision': scores['rougeL'].precision,
            'Recall': scores['rougeL'].recall,
            'F1-Score': scores['rougeL'].fmeasure
        }
    }

def summarize_text(text, model_name, max_length, min_length):
    """Generate summary using specified model"""
    summarizer = load_model(model_name)
    if summarizer is None:
        return None
    
    try:
        # Handle T5 models with prefix
        if model_name.startswith("t5"):
            text = f"summarize: {text}"
            result = summarizer(text, max_length=max_length, min_length=min_length)
            return result[0]['generated_text']
        else:
            result = summarizer(text, max_length=max_length, min_length=min_length, 
                              do_sample=False)
            return result[0]['summary_text']
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None

def create_rouge_visualization(rouge_scores):
    """Create visualization for ROUGE scores"""
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    precision_scores = [rouge_scores[metric]['Precision'] for metric in metrics]
    recall_scores = [rouge_scores[metric]['Recall'] for metric in metrics]
    f1_scores = [rouge_scores[metric]['F1-Score'] for metric in metrics]
    
    fig = go.Figure(data=[
        go.Bar(name='Precision', x=metrics, y=precision_scores),
        go.Bar(name='Recall', x=metrics, y=recall_scores),
        go.Bar(name='F1-Score', x=metrics, y=f1_scores)
    ])
    
    fig.update_layout(
        title='ROUGE Score Comparison',
        xaxis_title='ROUGE Metrics',
        yaxis_title='Score',
        barmode='group',
        height=400
    )
    
    return fig

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
model_options = {
    "BART (Recommended)": "facebook/bart-large-cnn",
    "T5 Small": "t5-small",
    "T5 Base": "t5-base",
    "DistilBART": "sshleifer/distilbart-cnn-12-6"
}

selected_model_name = st.sidebar.selectbox(
    "Choose Summarization Model:",
    list(model_options.keys())
)
model_name = model_options[selected_model_name]

# Summary parameters
st.sidebar.subheader("Summary Parameters")
max_length = st.sidebar.slider("Maximum Length", 50, 300, 150)
min_length = st.sidebar.slider("Minimum Length", 10, 100, 50)

# Application mode
st.sidebar.subheader("Application Mode")
app_mode = st.sidebar.selectbox(
    "Choose Mode:",
    ["Single Text Summarization", "Batch Processing", "Model Comparison"]
)

# Main application logic
if app_mode == "Single Text Summarization":
    st.header("📄 Single Text Summarization")
    
    # Input methods
    input_method = st.radio("Choose input method:", ["Text Area", "File Upload"])
    
    input_text = ""
    if input_method == "Text Area":
        input_text = st.text_area(
            "Enter text to summarize:",
            height=300,
            placeholder="Paste your article, research paper, or any long text here..."
        )
    else:
        uploaded_file = st.file_uploader("Upload text file", type=['txt'])
        if uploaded_file is not None:
            input_text = str(uploaded_file.read(), "utf-8")
            st.text_area("Uploaded text:", input_text, height=200)
    
    # Reference summary for evaluation
    reference_summary = st.text_area(
        "Reference Summary (optional - for ROUGE evaluation):",
        height=100,
        help="Provide a reference summary to calculate ROUGE scores"
    )
    
    if st.button("🚀 Generate Summary", type="primary"):
        if input_text.strip():
            with st.spinner(f"Generating summary using {selected_model_name}..."):
                start_time = time.time()
                summary = summarize_text(input_text, model_name, max_length, min_length)
                end_time = time.time()
                
                if summary:
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📋 Generated Summary")
                        st.write(summary)
                        
                        # Statistics
                        st.subheader("📊 Statistics")
                        original_words = len(input_text.split())
                        summary_words = len(summary.split())
                        compression_ratio = (1 - summary_words/original_words) * 100
                        
                        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                        stats_col1.metric("Original Words", original_words)
                        stats_col2.metric("Summary Words", summary_words)
                        stats_col3.metric("Compression", f"{compression_ratio:.1f}%")
                        stats_col4.metric("Processing Time", f"{end_time-start_time:.2f}s")
                    
                    with col2:
                        st.subheader("ℹ️ Model Info")
                        st.info(f"""
                        **Model:** {selected_model_name}
                        **Type:** {'Text-to-Text' if model_name.startswith('t5') else 'Summarization'}
                        **Max Length:** {max_length}
                        **Min Length:** {min_length}
                        """)
                    
                    # ROUGE evaluation if reference provided
                    if reference_summary.strip():
                        st.subheader("📏 ROUGE Evaluation")
                        rouge_scores = calculate_rouge_scores(summary, reference_summary)
                        
                        # Display ROUGE table
                        rouge_df = pd.DataFrame(rouge_scores).T
                        st.dataframe(rouge_df.round(4), use_container_width=True)
                        
                        # Visualization
                        fig = create_rouge_visualization(rouge_scores)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter text to summarize!")

elif app_mode == "Batch Processing":
    st.header("📊 Batch Processing")
    
    # File upload for batch processing
    uploaded_file = st.file_uploader(
        "Upload file with multiple texts",
        type=['json', 'csv', 'txt'],
        help="JSON: {'texts': ['text1', 'text2']}, CSV: column 'text', TXT: one text per line"
    )
    
    if uploaded_file is not None:
        file_type = uploaded_file.type
        
        try:
            if file_type == "application/json":
                data = json.load(uploaded_file)
                texts = data.get('texts', [])
            elif file_type == "text/csv":
                df = pd.read_csv(uploaded_file)
                texts = df['text'].tolist() if 'text' in df.columns else []
            else:  # txt file
                content = str(uploaded_file.read(), "utf-8")
                texts = [line.strip() for line in content.split('\n') if line.strip()]
            
            if texts:
                st.success(f"Loaded {len(texts)} texts for processing")
                
                if st.button("🚀 Process All Texts", type="primary"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, text in enumerate(texts):
                        progress_bar.progress((i + 1) / len(texts))
                        
                        with st.spinner(f"Processing text {i+1}/{len(texts)}..."):
                            summary = summarize_text(text, model_name, max_length, min_length)
                            
                            results.append({
                                'Text_ID': i+1,
                                'Original_Length': len(text.split()),
                                'Summary_Length': len(summary.split()) if summary else 0,
                                'Original_Text': text[:200] + "..." if len(text) > 200 else text,
                                'Summary': summary or "Error in processing"
                            })
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    st.subheader("📋 Processing Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="batch_summarization_results.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("📊 Batch Statistics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Texts", len(texts))
                    col2.metric("Avg Original Length", f"{results_df['Original_Length'].mean():.1f}")
                    col3.metric("Avg Summary Length", f"{results_df['Summary_Length'].mean():.1f}")
            else:
                st.error("No valid texts found in the uploaded file!")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif app_mode == "Model Comparison":
    st.header("⚖️ Model Comparison")
    
    # Select models to compare
    models_to_compare = st.multiselect(
        "Select models to compare:",
        list(model_options.keys()),
        default=["BART (Recommended)", "T5 Small"]
    )
    
    input_text = st.text_area(
        "Enter text for comparison:",
        height=200,
        placeholder="Enter text to compare summarization across different models..."
    )
    
    reference_summary = st.text_area(
        "Reference Summary (for ROUGE evaluation):",
        height=100
    )
    
    if st.button("🔍 Compare Models", type="primary") and input_text.strip():
        if len(models_to_compare) < 2:
            st.warning("Please select at least 2 models for comparison!")
        else:
            comparison_results = []
            
            for model_display_name in models_to_compare:
                model_path = model_options[model_display_name]
                
                with st.spinner(f"Processing with {model_display_name}..."):
                    start_time = time.time()
                    summary = summarize_text(input_text, model_path, max_length, min_length)
                    end_time = time.time()
                    
                    result = {
                        'Model': model_display_name,
                        'Summary': summary or "Error in processing",
                        'Processing_Time': f"{end_time-start_time:.2f}s",
                        'Word_Count': len(summary.split()) if summary else 0
                    }
                    
                    if reference_summary.strip() and summary:
                        rouge_scores = calculate_rouge_scores(summary, reference_summary)
                        result['ROUGE-1_F1'] = rouge_scores['ROUGE-1']['F1-Score']
                        result['ROUGE-2_F1'] = rouge_scores['ROUGE-2']['F1-Score']
                        result['ROUGE-L_F1'] = rouge_scores['ROUGE-L']['F1-Score']
                    
                    comparison_results.append(result)
            
            # Display comparison results
            st.subheader("📋 Model Comparison Results")
            
            for i, result in enumerate(comparison_results):
                with st.expander(f"{result['Model']} - {result['Word_Count']} words - {result['Processing_Time']}", 
                               expanded=True):
                    st.write(result['Summary'])
                    
                    if reference_summary.strip():
                        col1, col2, col3 = st.columns(3)
                        col1.metric("ROUGE-1 F1", f"{result.get('ROUGE-1_F1', 0):.3f}")
                        col2.metric("ROUGE-2 F1", f"{result.get('ROUGE-2_F1', 0):.3f}")
                        col3.metric("ROUGE-L F1", f"{result.get('ROUGE-L_F1', 0):.3f}")
            
            # Performance comparison chart
            if reference_summary.strip():
                st.subheader("📊 Performance Comparison")
                
                models = [r['Model'] for r in comparison_results]
                rouge1_scores = [r.get('ROUGE-1_F1', 0) for r in comparison_results]
                rouge2_scores = [r.get('ROUGE-2_F1', 0) for r in comparison_results]
                rougeL_scores = [r.get('ROUGE-L_F1', 0) for r in comparison_results]
                
                fig = go.Figure(data=[
                    go.Bar(name='ROUGE-1', x=models, y=rouge1_scores),
                    go.Bar(name='ROUGE-2', x=models, y=rouge2_scores),
                    go.Bar(name='ROUGE-L', x=models, y=rougeL_scores)
                ])
                
                fig.update_layout(
                    title='Model Performance Comparison (ROUGE F1 Scores)',
                    xaxis_title='Models',
                    yaxis_title='F1 Score',
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info("""
This AI Text Summarizer uses state-of-the-art transformer models:
- **BART**: Facebook's BART model fine-tuned for CNN/Daily Mail
- **T5**: Google's Text-to-Text Transfer Transformer
- **DistilBART**: Lightweight version of BART

**Features:**
- Single text summarization
- Batch processing capabilities
- Model performance comparison
- ROUGE score evaluation
- Customizable parameters
""")

st.sidebar.markdown("### 🔧 Technical Details")
st.sidebar.code(f"""
Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}
Models Cached: {len(st.session_state.get('models', {}))}
""")
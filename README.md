# AI-Powered Text Summarizer

## Project Overview

This project implements an AI-powered text summarizer using state-of-the-art transformer models. The application supports both extractive and abstractive summarization techniques with a user-friendly web interface built using Streamlit.

## Features

- **Multiple Model Support**: BART, T5, and other transformer models
- **Batch Processing**: Summarize multiple articles at once
- **Web Interface**: Interactive Streamlit dashboard
- **Model Comparison**: Compare different summarization approaches
- **Performance Metrics**: ROUGE score evaluation
- **Customizable Parameters**: Adjustable summary length and other parameters

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (recommended 8GB for larger models)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-text-summarizer.git
cd ai-text-summarizer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit==1.29.0
transformers==4.35.0
torch==2.1.0
tokenizers==0.14.1
rouge-score==0.1.2
nltk==3.8.1
spacy==3.7.2
pandas==2.1.3
numpy==1.24.3
plotly==5.17.0
requests==2.31.0
beautifulsoup4==4.12.2
```

## Project Structure

```
ai-text-summarizer/
├── app.py                 # Main Streamlit application
├── summarizer/
│   ├── __init__.py
│   ├── models.py          # Model implementations
│   ├── evaluator.py       # ROUGE score evaluation
│   └── utils.py           # Utility functions
├── data/
│   ├── sample_articles.txt
│   └── test_data.json
├── requirements.txt
├── README.md
└── setup.py
```

## Usage Guide

### Running the Application

1. Start the Streamlit web application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

### Using the Web Interface

1. **Single Text Summarization**:
   - Paste your text in the input area
   - Select model (BART, T5, etc.)
   - Adjust summary length
   - Click "Summarize"

2. **Batch Processing**:
   - Upload a text file or JSON file with multiple articles
   - Select processing options
   - Download results as CSV

3. **Model Comparison**:
   - Enable comparison mode
   - Select multiple models
   - View side-by-side results

### Command Line Usage

```python
from summarizer.models import TextSummarizer

# Initialize summarizer
summarizer = TextSummarizer(model_name="facebook/bart-large-cnn")

# Summarize text
text = "Your long text here..."
summary = summarizer.summarize(text, max_length=150)
print(summary)
```

## Model Details

### BART (Bidirectional and Auto-Regressive Transformers)

- **Model**: facebook/bart-large-cnn
- **Type**: Abstractive summarization
- **Strengths**: High-quality summaries, good coherence
- **Use Case**: News articles, general text

### T5 (Text-to-Text Transfer Transformer)

- **Model**: t5-small, t5-base
- **Type**: Text-to-text generation
- **Strengths**: Versatile, task-specific prompting
- **Use Case**: Various text types with custom prompts

### Model Selection Criteria

- **BART**: Best for news articles and formal text
- **T5**: More flexible for different text types
- **Performance**: BART generally produces higher quality summaries
- **Speed**: T5-small is faster but less accurate

## NLP Approach

### Preprocessing Pipeline

1. **Text Cleaning**: Remove HTML tags, special characters
2. **Tokenization**: Split text into tokens using model-specific tokenizers
3. **Chunking**: Handle long texts by splitting into manageable segments
4. **Encoding**: Convert text to model input format

### Summarization Process

1. **Input Processing**: Clean and tokenize input text
2. **Model Inference**: Generate summary using transformer model
3. **Post-processing**: Clean and format output summary
4. **Evaluation**: Calculate ROUGE scores if reference available

## Evaluation Metrics

### ROUGE Scores

- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap measurement
- **ROUGE-L**: Longest common subsequence based scoring

### Implementation

```python
from rouge_score import rouge_scorer

def evaluate_summary(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                     use_stemmer=True)
    scores = scorer.score(generated, reference)
    return scores
```

## API Documentation

### Core Functions

#### `TextSummarizer` Class

```python
class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        # Initialize model and tokenizer
        
    def summarize(self, text, max_length=150, min_length=50):
        # Generate summary
        
    def batch_summarize(self, texts, **kwargs):
        # Process multiple texts
```

#### `evaluate_summary` Function

```python
def evaluate_summary(generated_summary, reference_summary):
    # Calculate ROUGE scores
    return rouge_scores
```

## Performance Optimization

### Memory Management

- Use model checkpointing for large texts
- Implement batch processing with appropriate batch sizes
- Clear GPU memory after processing

### Speed Optimization

- Use smaller models for faster inference
- Implement caching for repeated summarizations
- Use CPU for small texts, GPU for large batches

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce batch size
   - Use smaller model variants
   - Process texts in chunks

2. **Slow Performance**:
   - Check GPU availability
   - Reduce max_length parameter
   - Use lighter models

3. **Poor Summary Quality**:
   - Ensure input text is well-formatted
   - Try different models
   - Adjust length parameters

### Error Handling

The application includes comprehensive error handling for:
- Invalid input formats
- Model loading failures
- Memory limitations
- Network connectivity issues

## Advanced Features

### Custom Model Fine-tuning

```python
# Example fine-tuning setup
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    evaluation_strategy="steps",
)
```

### Multi-language Support

- Supports models trained on multiple languages
- Automatic language detection
- Language-specific preprocessing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- Facebook AI Research for BART model
- Google Research for T5 model
- ROUGE evaluation metric implementation

## Contact

For questions or support, please contact [your-email@example.com]

---

## Quick Start Example

```python
# Install dependencies
pip install streamlit transformers torch rouge-score

# Create simple summarizer
from transformers import pipeline

summarizer = pipeline("summarization", 
                     model="facebook/bart-large-cnn")

text = """
The Inflation Reduction Act lowers prescription drug costs, 
health care costs, and energy costs. It's the most aggressive 
action on tackling the climate crisis in American history, 
which will lift up American workers and create good-paying, 
union jobs across the country.
"""

summary = summarizer(text, max_length=50, min_length=25, 
                    do_sample=False)
print(summary[0]['summary_text'])
```

This comprehensive documentation provides everything needed to understand, setup, and use the AI-powered text summarizer effectively.

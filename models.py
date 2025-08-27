from abc import ABC, abstractmethod
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseSummarizer(ABC):
    """Abstract base class for text summarizers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer"""
        pass
    
    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text before summarization"""
        pass
    
    @abstractmethod
    def postprocess_summary(self, summary: str) -> str:
        """Postprocess generated summary"""
        pass
    
    @abstractmethod
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Generate summary from input text"""
        pass

class BARTSummarizer(BaseSummarizer):
    """BART model for abstractive summarization"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        super().__init__(model_name)
        self.load_model()
    
    def load_model(self):
        """Load BART model and tokenizer"""
        try:
            self.model = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device
            )
            logger.info(f"Successfully loaded BART model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading BART model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Clean and prepare text for BART"""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Handle very long texts by truncating
        max_input_length = 1024
        words = text.split()
        if len(words) > max_input_length:
            text = ' '.join(words[:max_input_length])
            logger.warning(f"Text truncated to {max_input_length} words")
        
        return text
    
    def postprocess_summary(self, summary: str) -> str:
        """Clean up generated summary"""
        # Remove extra whitespace
        summary = ' '.join(summary.split())
        
        # Ensure proper sentence ending
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Generate summary using BART"""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess input
            processed_text = self.preprocess_text(text)
            
            # Generate summary
            result = self.model(
                processed_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            # Postprocess and return
            summary = result[0]['summary_text']
            return self.postprocess_summary(summary)
            
        except Exception as e:
            logger.error(f"Error during BART summarization: {str(e)}")
            return f"Error: {str(e)}"

class T5Summarizer(BaseSummarizer):
    """T5 model for text-to-text summarization"""
    
    def __init__(self, model_name: str = "t5-small"):
        super().__init__(model_name)
        self.load_model()
    
    def load_model(self):
        """Load T5 model and tokenizer"""
        try:
            self.model = pipeline(
                "text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device
            )
            logger.info(f"Successfully loaded T5 model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading T5 model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Prepare text for T5 with task prefix"""
        # Add task prefix for T5
        text = f"summarize: {text}"
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Handle long texts
        max_input_length = 512  # T5 has shorter context
        words = text.split()
        if len(words) > max_input_length:
            text = ' '.join(words[:max_input_length])
            logger.warning(f"Text truncated to {max_input_length} words")
        
        return text
    
    def postprocess_summary(self, summary: str) -> str:
        """Clean up T5 generated summary"""
        # Remove extra whitespace
        summary = ' '.join(summary.split())
        
        # Capitalize first letter
        if summary:
            summary = summary[0].upper() + summary[1:] if len(summary) > 1 else summary.upper()
        
        # Ensure proper sentence ending
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Generate summary using T5"""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess input
            processed_text = self.preprocess_text(text)
            
            # Generate summary
            result = self.model(
                processed_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            # Postprocess and return
            summary = result[0]['generated_text']
            return self.postprocess_summary(summary)
            
        except Exception as e:
            logger.error(f"Error during T5 summarization: {str(e)}")
            return f"Error: {str(e)}"

class TextSummarizer:
    """Main text summarizer class that manages different models"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.summarizer = self._initialize_summarizer()
    
    def _initialize_summarizer(self) -> BaseSummarizer:
        """Initialize the appropriate summarizer based on model name"""
        if "bart" in self.model_name.lower():
            return BARTSummarizer(self.model_name)
        elif "t5" in self.model_name.lower():
            return T5Summarizer(self.model_name)
        else:
            # Default to BART for unknown models
            logger.warning(f"Unknown model type, defaulting to BART: {self.model_name}")
            return BARTSummarizer(self.model_name)
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Generate summary using the configured model"""
        return self.summarizer.summarize(text, max_length, min_length)
    
    def batch_summarize(self, texts: list, max_length: int = 150, min_length: int = 50) -> list:
        """Summarize multiple texts"""
        summaries = []
        for i, text in enumerate(texts):
            try:
                summary = self.summarize(text, max_length, min_length)
                summaries.append(summary)
                logger.info(f"Successfully summarized text {i+1}/{len(texts)}")
            except Exception as e:
                logger.error(f"Error summarizing text {i+1}: {str(e)}")
                summaries.append(f"Error: {str(e)}")
        
        return summaries
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "model_type": type(self.summarizer).__name__,
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "max_input_length": 1024 if "bart" in self.model_name.lower() else 512
        }

# Model factory function
def create_summarizer(model_name: str) -> TextSummarizer:
    """Factory function to create a text summarizer"""
    return TextSummarizer(model_name)

# Available models configuration
AVAILABLE_MODELS = {
    "facebook/bart-large-cnn": {
        "name": "BART Large CNN",
        "type": "Abstractive",
        "description": "Facebook's BART model fine-tuned on CNN/Daily Mail dataset",
        "best_for": "News articles, formal text",
        "max_length": 1024
    },
    "facebook/bart-large": {
        "name": "BART Large",
        "type": "Abstractive", 
        "description": "Base BART model for general summarization",
        "best_for": "General text summarization",
        "max_length": 1024
    },
    "t5-small": {
        "name": "T5 Small",
        "type": "Text-to-Text",
        "description": "Google's T5 small model for text generation",
        "best_for": "Quick summarization, resource-constrained environments",
        "max_length": 512
    },
    "t5-base": {
        "name": "T5 Base",
        "type": "Text-to-Text",
        "description": "Google's T5 base model for text generation",
        "best_for": "Balanced performance and quality",
        "max_length": 512
    },
    "sshleifer/distilbart-cnn-12-6": {
        "name": "DistilBART CNN",
        "type": "Abstractive",
        "description": "Distilled version of BART for faster inference",
        "best_for": "Fast summarization with good quality",
        "max_length": 1024
    }
}

def get_available_models() -> dict:
    """Get dictionary of available models with their configurations"""
    return AVAILABLE_MODELS

def get_model_recommendations(text_length: int, speed_priority: bool = False) -> list:
    """Get model recommendations based on text characteristics"""
    recommendations = []
    
    if speed_priority:
        recommendations.extend([
            "sshleifer/distilbart-cnn-12-6",
            "t5-small"
        ])
    
    if text_length > 1000:
        recommendations.extend([
            "facebook/bart-large-cnn",
            "facebook/bart-large"
        ])
    else:
        recommendations.extend([
            "t5-base",
            "facebook/bart-large-cnn"
        ])
    
    return list(dict.fromkeys(recommendations))  # Remove duplicates while preserving order
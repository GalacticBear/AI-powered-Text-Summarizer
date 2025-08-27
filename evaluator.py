from rouge_score import rouge_scorer
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ROUGEEvaluator:
    """Class for evaluating summaries using ROUGE metrics"""
    
    def __init__(self, use_stemmer: bool = True):
        """
        Initialize ROUGE evaluator
        
        Args:
            use_stemmer: Whether to use stemming in ROUGE calculation
        """
        self.use_stemmer = use_stemmer
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=use_stemmer
        )
    
    def evaluate_single(self, generated_summary: str, reference_summary: str) -> Dict:
        """
        Evaluate a single generated summary against reference
        
        Args:
            generated_summary: Model-generated summary
            reference_summary: Human-written reference summary
            
        Returns:
            Dictionary containing ROUGE scores
        """
        try:
            scores = self.scorer.score(generated_summary, reference_summary)
            
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
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {str(e)}")
            return self._empty_scores()
    
    def evaluate_batch(self, generated_summaries: List[str], 
                      reference_summaries: List[str]) -> Dict:
        """
        Evaluate multiple summaries and return aggregate statistics
        
        Args:
            generated_summaries: List of model-generated summaries
            reference_summaries: List of reference summaries
            
        Returns:
            Dictionary containing mean ROUGE scores and statistics
        """
        if len(generated_summaries) != len(reference_summaries):
            raise ValueError("Number of generated and reference summaries must match")
        
        all_scores = []
        
        for gen_summary, ref_summary in zip(generated_summaries, reference_summaries):
            scores = self.evaluate_single(gen_summary, ref_summary)
            all_scores.append(scores)
        
        return self._calculate_aggregate_scores(all_scores)
    
    def _calculate_aggregate_scores(self, all_scores: List[Dict]) -> Dict:
        """Calculate mean and standard deviation of ROUGE scores"""
        rouge_types = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        metric_types = ['Precision', 'Recall', 'F1-Score']
        
        aggregate_scores = {}
        
        for rouge_type in rouge_types:
            aggregate_scores[rouge_type] = {}
            
            for metric_type in metric_types:
                values = [scores[rouge_type][metric_type] for scores in all_scores 
                         if rouge_type in scores and metric_type in scores[rouge_type]]
                
                if values:
                    aggregate_scores[rouge_type][metric_type] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                else:
                    aggregate_scores[rouge_type][metric_type] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0
                    }
        
        return aggregate_scores
    
    def _empty_scores(self) -> Dict:
        """Return empty ROUGE scores structure"""
        return {
            'ROUGE-1': {'Precision': 0.0, 'Recall': 0.0, 'F1-Score': 0.0},
            'ROUGE-2': {'Precision': 0.0, 'Recall': 0.0, 'F1-Score': 0.0},
            'ROUGE-L': {'Precision': 0.0, 'Recall': 0.0, 'F1-Score': 0.0}
        }
    
    def compare_models(self, model_summaries: Dict[str, List[str]], 
                      reference_summaries: List[str]) -> Dict:
        """
        Compare multiple models' performance using ROUGE scores
        
        Args:
            model_summaries: Dictionary mapping model names to their summaries
            reference_summaries: List of reference summaries
            
        Returns:
            Dictionary containing comparison results
        """
        comparison_results = {}
        
        for model_name, summaries in model_summaries.items():
            try:
                scores = self.evaluate_batch(summaries, reference_summaries)
                comparison_results[model_name] = scores
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {str(e)}")
                comparison_results[model_name] = None
        
        return comparison_results
    
    def get_best_model(self, comparison_results: Dict, 
                      metric: str = 'F1-Score', rouge_type: str = 'ROUGE-1') -> Tuple[str, float]:
        """
        Identify the best performing model based on specified metric
        
        Args:
            comparison_results: Results from compare_models()
            metric: Metric to use for comparison (Precision, Recall, F1-Score)
            rouge_type: ROUGE type to use (ROUGE-1, ROUGE-2, ROUGE-L)
            
        Returns:
            Tuple of (best_model_name, best_score)
        """
        best_model = None
        best_score = -1
        
        for model_name, results in comparison_results.items():
            if results and rouge_type in results and metric in results[rouge_type]:
                score = results[rouge_type][metric]['mean']
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model, best_score

class QualityMetrics:
    """Additional quality metrics for text summarization evaluation"""
    
    @staticmethod
    def compression_ratio(original_text: str, summary: str) -> float:
        """
        Calculate compression ratio (1 - summary_length/original_length)
        
        Args:
            original_text: Original input text
            summary: Generated summary
            
        Returns:
            Compression ratio as float
        """
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        
        if original_words == 0:
            return 0.0
        
        return 1 - (summary_words / original_words)
    
    @staticmethod
    def extractive_fragments(original_text: str, summary: str, 
                           min_fragment_length: int = 4) -> float:
        """
        Calculate percentage of summary that consists of extractive fragments
        
        Args:
            original_text: Original input text
            summary: Generated summary
            min_fragment_length: Minimum length of fragments to consider
            
        Returns:
            Percentage of extractive content
        """
        original_words = original_text.lower().split()
        summary_words = summary.lower().split()
        
        if not summary_words:
            return 0.0
        
        extractive_words = 0
        
        for i in range(len(summary_words)):
            for j in range(i + min_fragment_length, len(summary_words) + 1):
                fragment = summary_words[i:j]
                fragment_str = ' '.join(fragment)
                
                if fragment_str in ' '.join(original_words):
                    extractive_words += len(fragment)
                    break
        
        return (extractive_words / len(summary_words)) * 100
    
    @staticmethod
    def novel_ngrams(original_text: str, summary: str, n: int = 2) -> float:
        """
        Calculate percentage of novel n-grams in summary
        
        Args:
            original_text: Original input text
            summary: Generated summary
            n: N-gram size
            
        Returns:
            Percentage of novel n-grams
        """
        def get_ngrams(text: str, n: int) -> set:
            words = text.lower().split()
            return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))
        
        original_ngrams = get_ngrams(original_text, n)
        summary_ngrams = get_ngrams(summary, n)
        
        if not summary_ngrams:
            return 0.0
        
        novel_ngrams = summary_ngrams - original_ngrams
        return (len(novel_ngrams) / len(summary_ngrams)) * 100

def evaluate_summary_quality(original_text: str, generated_summary: str, 
                           reference_summary: str = None) -> Dict:
    """
    Comprehensive evaluation of summary quality
    
    Args:
        original_text: Original input text
        generated_summary: Model-generated summary
        reference_summary: Optional reference summary for ROUGE evaluation
        
    Returns:
        Dictionary containing various quality metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['statistics'] = {
        'original_word_count': len(original_text.split()),
        'summary_word_count': len(generated_summary.split()),
        'compression_ratio': QualityMetrics.compression_ratio(original_text, generated_summary)
    }
    
    # Quality metrics
    metrics['quality'] = {
        'extractive_fragments_pct': QualityMetrics.extractive_fragments(
            original_text, generated_summary
        ),
        'novel_bigrams_pct': QualityMetrics.novel_ngrams(
            original_text, generated_summary, n=2
        ),
        'novel_trigrams_pct': QualityMetrics.novel_ngrams(
            original_text, generated_summary, n=3
        )
    }
    
    # ROUGE evaluation if reference provided
    if reference_summary:
        evaluator = ROUGEEvaluator()
        metrics['rouge_scores'] = evaluator.evaluate_single(
            generated_summary, reference_summary
        )
    
    return metrics
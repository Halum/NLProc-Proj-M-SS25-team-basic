from bert_score import score as bert_score
from rouge_score import rouge_scorer


class MetricsGenerator:
    def __init__(self, insight_df):
        """
        Initialize the MetricsGenerator with the insights DataFrame.
        
        Args:
            insight_df (pd.DataFrame): DataFrame containing insights from evaluations
        """
        pass
    
    @staticmethod
    def calculate_bert_score(gold_answers: str, generated_answers: str) -> dict:
        """
        Calculate BERT score for the generated answers against gold answers.
        
        Returns:
            pd.DataFrame: DataFrame with BERT scores added
        """
        
        # Calculate BERT scores
        P, R, F1 = bert_score([generated_answers], [gold_answers], lang='en', model_type="bert-base-uncased", verbose=True)

        bert_scores = {
            'bert_precision': P.item(),
            'bert_recall': R.item(),
            'bert_f1': F1.item()
        }
        
        print(f"BERT scores calculated: {bert_scores}")

        return bert_scores
    
    @staticmethod
    def calculate_rouge_score(gold_answers: str, generated_answers: str) -> dict:
        """
        Calculate ROUGE score for the generated answers against gold answers.
        
        Returns:
            dict: Dictionary with ROUGE scores
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(gold_answers, generated_answers)
        
        rouge_scores = {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_fmeasure': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_fmeasure': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_fmeasure': scores['rougeL'].fmeasure
        }

        return rouge_scores
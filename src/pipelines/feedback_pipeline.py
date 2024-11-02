from typing import Dict, Any, Optional
import pandas as pd
from generators.feedback_generator import FeedbackGenerator

def generate_feedback_pipeline(
    model_name: str,
    anthropic_key: str,
    wrong_answers_df: pd.DataFrame,
    batch_size: int = 5,
    output_file: Optional[str] = None
) -> Dict[str, Dict[str, str]]:
    """
    Pipeline to generate feedback for wrong answers using the FeedbackGenerator class.

    Args:
        model_name (str): Name of the AI model to use for feedback generation.
        anthropic_key (str): Anthropic API key.
        wrong_answers_df (pd.DataFrame): DataFrame containing wrong answers with required columns.
        batch_size (int): Number of samples to process in each batch.
        output_file (str, optional): Path to save the generated feedback as a CSV file.

    Returns:
        Dict[str, Dict[str, str]]: Dictionary containing the generated feedback.
    """
    # Initialize FeedbackGenerator
    feedback_gen = FeedbackGenerator(
        model_name=model_name,
        anthropic_key=anthropic_key
    )
    
    # Generate Feedback
    feedback_data = feedback_gen.generate_feedback(
        wrong_answers_df=wrong_answers_df,
        batch_size=batch_size
    )
    
    # Optionally, save the feedback to a CSV file
    if output_file:
        feedback_gen.save_feedback_to_csv(feedback_data, output_file)
    
    return feedback_data 
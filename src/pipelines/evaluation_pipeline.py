from typing import Optional, Dict, Any, Tuple
import pandas as pd
from evaluation.answer_comparator import AnswerComparator
from training.dpo_training import DPOTrainingPipeline
from tqdm import tqdm

def evaluate_dpo_model_pipeline(
    model_path: str,
    df: pd.DataFrame,
    question_col: str,
    answer_col: str,
    device: str = "cuda",
    max_seq_length: int = 2048,
    with_confidence: bool = False,
    mode: str = "cot",
    system_prompt: Optional[str] = None,
    batch_size: int = 4
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Pipeline to evaluate a DPO-trained model's responses using AnswerComparator.

    Args:
        model_path: Path to the trained DPO model.
        df: DataFrame containing questions and answers.
        question_col: Column name for questions.
        answer_col: Column name for correct answers.
        device: Device to run the model on.
        max_seq_length: Maximum sequence length for the model.
        with_confidence: Whether to include confidence scores.
        mode: Reasoning mode ('cot' or 'tir').
        system_prompt: Optional custom system prompt.
        batch_size: Batch size for processing.

    Returns:
        Tuple of (DataFrame with evaluation results, metrics dictionary)
    """
    # Load the DPO-trained model
    print("Loading DPO model...")
    model, tokenizer = DPOTrainingPipeline.load_from_pretrained(
        model_path=model_path,
        device=device,
        max_seq_length=max_seq_length,
        load_in_4bit=True  # Use 4-bit quantization for efficiency
    )
    
    # Initialize DPO pipeline with proper configuration
    dpo_pipeline = DPOTrainingPipeline(
        mode=mode,
        max_seq_length=max_seq_length,
        batch_size=batch_size
    )
    
    # Generate responses for all questions in batches
    print("Generating model responses...")
    responses = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        batch_responses = []
        
        for _, row in batch_df.iterrows():
            try:
                response = dpo_pipeline.generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    question=str(row[question_col]),
                    mode=mode
                )
                batch_responses.append(response)
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                batch_responses.append("")
                
        responses.extend(batch_responses)
    
    # Add responses to dataframe
    results_df = df.copy()
    results_df['model_response'] = responses
    
    # Initialize the answer comparator with smaller evaluation model
    print("Evaluating responses...")
    comparator = AnswerComparator(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device,
        system_prompt=system_prompt
    )
    
    # Process the dataframe with the comparator
    evaluation_df = comparator.process_dataframe(
        df=results_df,
        question_col=question_col,
        response_col='model_response',
        answer_col=answer_col,
        with_confidence=with_confidence
    )
    
    # Calculate metrics
    metrics = comparator.calculate_metrics(evaluation_df)
    print("Evaluation Metrics:", metrics)
    
    return evaluation_df, metrics 
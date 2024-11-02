from typing import Optional, Dict, Any, Tuple
import pandas as pd
from evaluation.answer_comparator import AnswerComparator
from training.dpo_training import DPOTrainingPipeline
from tqdm import tqdm

def evaluate_model_pipeline(
    df: pd.DataFrame,
    question_col: str,
    answer_col: str,
    response_col: Optional[str] = None,
    model_path: Optional[str] = None,
    device: str = "cuda",
    max_seq_length: int = 2048,
    with_confidence: bool = False,
    mode: str = "cot",
    system_prompt: Optional[str] = None,
    batch_size: int = 4,
    is_dpo_model: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Pipeline to evaluate model responses using AnswerComparator.
    Can either evaluate existing responses or generate and evaluate new ones.

    Args:
        df: DataFrame containing questions and answers
        question_col: Column name for questions
        answer_col: Column name for correct answers
        response_col: Optional column name containing existing model responses
        model_path: Optional path to model for generating responses
        device: Device to run the model on
        max_seq_length: Maximum sequence length for the model
        with_confidence: Whether to include confidence scores
        mode: Reasoning mode ('cot' or 'tir')
        system_prompt: Optional custom system prompt
        batch_size: Batch size for processing
        is_dpo_model: Whether the model is DPO-trained

    Returns:
        Tuple of (DataFrame with evaluation results, metrics dictionary)
    """
    results_df = df.copy()
    
    # Generate responses if needed
    if response_col is None:
        if model_path is None:
            raise ValueError("Either response_col or model_path must be provided")
            
        print("Generating model responses...")
        if is_dpo_model:
            # Use existing DPO model generation code
            model, tokenizer = DPOTrainingPipeline.load_from_pretrained(
                model_path=model_path,
                device=device,
                max_seq_length=max_seq_length,
                load_in_4bit=True
            )
            
            dpo_pipeline = DPOTrainingPipeline(
                mode=mode,
                max_seq_length=max_seq_length,
                batch_size=batch_size
            )
            
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
                
        else:
            # Import and use base model execution
            from pipelines.execution_pipeline import execute_model_pipeline
            temp_df = execute_model_pipeline(
                df=df,
                input_col=question_col,
                model_path=model_path,
                mode=mode,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                device=device
            )
            responses = temp_df['model_response'].tolist()
            
        results_df['model_response'] = responses
        response_col = 'model_response'

    # Initialize the answer comparator
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
        response_col=response_col,
        answer_col=answer_col,
        with_confidence=with_confidence
    )
    
    # Calculate metrics
    metrics = comparator.calculate_metrics(evaluation_df)
    print("Evaluation Metrics:", metrics)
    
    return evaluation_df, metrics
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from evaluation.answer_comparator import AnswerComparator
from training.dpo_training import DPOTrainingPipeline
from tqdm import tqdm
from handlers.reward_model_handler import RewardModelHandler

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
    is_dpo_model: bool = False,
    use_best_of_n: bool = False,
    n_samples: int = 5
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
        use_best_of_n: Whether to use best-of-n sampling
        n_samples: Number of samples to generate for best-of-n sampling

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
            
            if use_best_of_n:
                reward_model = RewardModelHandler(device=device)
                responses = []
                
                for i in tqdm(range(0, len(df), batch_size)):
                    batch_df = df.iloc[i:i + batch_size]
                    batch_questions = batch_df[question_col].tolist()
                    
                    batch_responses = dpo_pipeline.generate_with_best_of_n(
                        questions=batch_questions,
                        n_samples=n_samples,
                        reward_model=reward_model
                    )
                    
                    responses.extend(batch_responses)
            
        else:
            # Import and use base model execution
            from pipelines.execution_pipeline import execute_model_pipeline
            
            if use_best_of_n:
                reward_model = RewardModelHandler(device=device)
                responses = []
                
                for i in tqdm(range(0, len(df), batch_size)):
                    batch_df = df.iloc[i:i + batch_size]
                    batch_questions = batch_df[question_col].tolist()
                    
                    # Generate multiple responses for each question
                    all_responses = []
                    for _ in range(n_samples):
                        temp_df = execute_model_pipeline(
                            df=batch_df,
                            input_col=question_col,
                            model_path=model_path,
                            mode=mode,
                            batch_size=batch_size,
                            max_seq_length=max_seq_length,
                            device=device
                        )
                        all_responses.append(temp_df['model_response'].tolist())
                    
                    # Transpose to group responses by question
                    question_responses = list(zip(*all_responses))
                    
                    # Select best response for each question using reward model
                    batch_best_responses = []
                    for question_response_set in question_responses:
                        scores = reward_model.score_responses(question_response_set)
                        best_response = question_response_set[scores.argmax()]
                        batch_best_responses.append(best_response)
                    
                    responses.extend(batch_best_responses)
            else:
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
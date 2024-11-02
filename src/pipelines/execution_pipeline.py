from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import torch # type: ignore
from transformers import AutoTokenizer
from training.dpo_training import DPOTrainingPipeline
from handlers.model_handler import ModelHandler
from datasets import Dataset
from tqdm import tqdm

def execute_model_pipeline(
    df: pd.DataFrame,
    input_col: str,
    output_col: str = "model_response",
    id_col: Optional[str] = None,
    model_path: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    mode: str = "cot",
    batch_size: int = 4,
    max_seq_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    is_dpo_model: bool = False,
    device: str = "cuda",
    load_in_4bit: bool = True
) -> pd.DataFrame:
    """
    Pipeline for executing models and getting responses using DataFrame input.

    Args:
        df: Input DataFrame containing questions
        input_col: Column name containing input questions
        output_col: Column name where to store model responses
        id_col: Optional column name for question IDs
        model_path: Path to the model or model name
        mode: Reasoning mode ('cot' or 'tir')
        batch_size: Size of batches for processing
        max_seq_length: Maximum sequence length for the model
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        is_dpo_model: Whether the model is DPO-trained
        device: Device to run the model on
        load_in_4bit: Whether to load model in 4-bit precision

    Returns:
        DataFrame with added model responses
    """
    # Validate input
    if input_col not in df.columns:
        raise ValueError(f"Input column '{input_col}' not found in DataFrame")
    
    # Create ID column if not provided
    if id_col is None:
        df['temp_id'] = [f"q{i}" for i in range(len(df))]
        id_col = 'temp_id'
    
    # Create copy of DataFrame
    result_df = df.copy()
    result_df[output_col] = ""

    if is_dpo_model:
        responses = _execute_dpo_model(
            questions=df[input_col].tolist(),
            question_ids=df[id_col].tolist(),
            model_path=model_path,
            mode=mode,
            max_seq_length=max_seq_length,
            temperature=temperature,
            top_p=top_p,
            device=device,
            load_in_4bit=load_in_4bit
        )
    else:
        responses = _execute_base_model(
            questions=df[input_col].tolist(),
            question_ids=df[id_col].tolist(),
            model_path=model_path,
            mode=mode,
            batch_size=batch_size
        )

    # Update DataFrame with responses
    for response in responses:
        idx = result_df[result_df[id_col] == response["id"]].index[0]
        result_df.at[idx, output_col] = response["response"]

    # Remove temporary ID column if it was created
    if 'temp_id' in result_df.columns:
        result_df = result_df.drop('temp_id', axis=1)

    return result_df

def _execute_dpo_model(
    questions: List[str],
    question_ids: List[str],
    model_path: str,
    mode: str,
    max_seq_length: int,
    temperature: float,
    top_p: float,
    device: str,
    load_in_4bit: bool
) -> List[Dict[str, Any]]:
    """Handle execution for DPO-trained models."""
    # Load DPO model and tokenizer
    model, tokenizer = DPOTrainingPipeline.load_from_pretrained(
        model_path=model_path,
        device=device,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit
    )

    # Initialize DPO pipeline
    dpo_pipeline = DPOTrainingPipeline(
        mode=mode,
        max_seq_length=max_seq_length
    )

    # Generate responses
    responses = []
    for qid, question in tqdm(zip(question_ids, questions), total=len(questions)):
        try:
            response = dpo_pipeline.generate_response(
                model=model,
                tokenizer=tokenizer,
                question=question,
                mode=mode,
                temperature=temperature,
                top_p=top_p
            )
            responses.append({
                "id": qid,
                "question": question,
                "response": response,
                "status": "success"
            })
        except Exception as e:
            responses.append({
                "id": qid,
                "question": question,
                "response": "",
                "status": "error",
                "error": str(e)
            })

    return responses

def _execute_base_model(
    questions: List[str],
    question_ids: List[str],
    model_path: str,
    mode: str,
    batch_size: int
) -> List[Dict[str, Any]]:
    """Handle execution for base models using ModelHandler."""
    # Initialize model handler
    model_handler = ModelHandler(model_name=model_path)

    # Create batch inputs
    batched_inputs = model_handler.create_batch_inputs(
        questions=questions,
        ids=question_ids,
        batch_size=batch_size,
        mode=mode
    )

    # Generate responses
    responses = model_handler.generate_responses(batched_inputs)
    
    return responses 
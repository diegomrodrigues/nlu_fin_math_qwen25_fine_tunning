from typing import Optional
import pandas as pd
from ..training.dpo_training import DPOTrainingPipeline

def train_dpo_pipeline(
    model_name: str = "unsloth/Qwen2.5-Math-1.5B-bnb-4bit",
    original_model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    max_seq_length: int = 2048,
    output_dir: str = "./dpo_output",
    beta: float = 0.1,
    batch_size: int = 4,
    num_train_epochs: int = 3,
    learning_rate: float = 5e-5,
    mode: str = "cot",
    training_data_df: Optional[pd.DataFrame] = None,
    question_col: str = "QUESTION",
    chosen_col: str = "CORRECTED_RESPONSE",
    rejected_col: str = "AI_RESPONSE"
) -> None:
    """
    Pipeline to train a model using Direct Preference Optimization (DPO) with the DPOTrainingPipeline class.

    Args:
        model_name (str): Name of the model to fine-tune.
        original_model_name (str): Name of the original base model.
        max_seq_length (int): Maximum sequence length for the tokenizer.
        output_dir (str): Directory to save the trained model.
        beta (float): Hyperparameter for DPO.
        batch_size (int): Batch size for training.
        num_train_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        mode (str): Reasoning mode ('cot' or 'tir').
        training_data_df (pd.DataFrame, optional): DataFrame containing training data.
        question_col (str): Column name for questions in the DataFrame.
        chosen_col (str): Column name for chosen (correct) responses in the DataFrame.
        rejected_col (str): Column name for rejected (incorrect) responses in the DataFrame.

    Returns:
        None
    """
    # Initialize DPOTrainingPipeline
    dpo_pipeline = DPOTrainingPipeline(
        model_name=model_name,
        original_model_name=original_model_name,
        max_seq_length=max_seq_length,
        output_dir=output_dir,
        beta=beta,
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        mode=mode
    )
    
    # Prepare Training Data
    if training_data_df is None:
        raise ValueError("Training data DataFrame must be provided.")
    
    train_dataset = dpo_pipeline.prepare_training_data(
        df=training_data_df,
        question_col=question_col,
        chosen_col=chosen_col,
        rejected_col=rejected_col
    )
    
    # Train the Model
    dpo_pipeline.train(train_dataset=train_dataset)
    
    print("DPO training pipeline completed successfully.") 
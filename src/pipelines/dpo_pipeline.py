from typing import Optional, List
import pandas as pd
from training.dpo_training import DPOTrainingPipeline
import wandb # type: ignore

def train_dpo_pipeline(
    project_name: str,
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
    rejected_col: str = "AI_RESPONSE",
    gradient_accumulation_steps: int = 2,
    warmup_ratio: float = 0.1,
    lora_r: int = 4,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
    lora_target_modules: List[str] = ["up_proj", "down_proj"],
    use_gradient_checkpointing: bool = True,
    fp16: bool = True,
    optim: str = "paged_adamw_32bit",
    save_strategy: str = "epoch",
    random_state: int = 1337
) -> None:
    """
    Pipeline to train a model using Direct Preference Optimization (DPO) with the DPOTrainingPipeline class.

    Args:
        project_name (str): Name for the project (used for output_dir, wandb, and HuggingFace)
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
        gradient_accumulation_steps (int): Number of steps for gradient accumulation
        warmup_ratio (float): Ratio of warmup steps to total steps
        lora_r (int): LoRA rank
        lora_alpha (int): LoRA alpha parameter
        lora_dropout (float): LoRA dropout rate
        lora_target_modules (List[str]): Target modules for LoRA
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing
        fp16 (bool): Whether to use FP16 training
        optim (str): Optimizer type
        save_strategy (str): When to save model checkpoints
        random_state (int): Random seed for reproducibility

    Returns:
        None
    """
    # Update output directory based on project name
    output_dir = f"./dpo_output/{project_name}"
    
    # Initialize wandb
    wandb.init(project=project_name, name=project_name)
    
    # Initialize DPOTrainingPipeline with updated output_dir
    dpo_pipeline = DPOTrainingPipeline(
        model_name=model_name,
        original_model_name=original_model_name,
        max_seq_length=max_seq_length,
        output_dir=output_dir,
        beta=beta,
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        mode=mode,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        use_gradient_checkpointing=use_gradient_checkpointing,
        fp16=fp16,
        optim=optim,
        save_strategy=save_strategy,
        random_state=random_state,
        project_name=project_name
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
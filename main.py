import pandas as pd
from src.pipelines.question_pipeline import generate_questions_pipeline
from src.pipelines.feedback_pipeline import generate_feedback_pipeline
from src.pipelines.dpo_pipeline import train_dpo_pipeline

def main():
    # Example usage for generating questions
    generated_questions = generate_questions_pipeline(
        model_name="claude-sonnet",
        anthropic_key="your-anthropic-api-key",
        dataset_name="dreamerdeo/finqa",
        split="train",
        batch_size=8,
        total_samples=100,
        output_file="generated_questions.json"
    )
    
    # Load wrong answers DataFrame (replace with your actual data)
    wrong_answers = pd.read_csv("wrong_answers.csv")  # Ensure this file exists with required columns
    
    # Example usage for generating feedback
    feedback = generate_feedback_pipeline(
        model_name="claude-sonnet",
        anthropic_key="your-anthropic-api-key",
        wrong_answers_df=wrong_answers,  # DataFrame containing wrong answers
        batch_size=5,
        output_file="feedback.csv"
    )
    
    # Load training data DataFrame (replace with your actual data)
    training_data_df = pd.read_csv("training_data.csv")  # Ensure this file exists with required columns
    
    # Example usage for training via DPO
    train_dpo_pipeline(
        model_name="unsloth/Qwen2.5-Math-1.5B-bnb-4bit",
        original_model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        max_seq_length=2048,
        output_dir="./dpo_output",
        beta=0.1,
        batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        mode="cot",
        training_data_df=training_data_df,  # DataFrame containing training data
        question_col="QUESTION",
        chosen_col="CORRECTED_RESPONSE",
        rejected_col="AI_RESPONSE"
    )

if __name__ == "__main__":
    main() 
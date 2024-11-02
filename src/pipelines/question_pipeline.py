from typing import Dict, Any, Optional
import json
from generators.question_generator import QuestionGenerator

def generate_questions_pipeline(
    model_name: str,
    anthropic_key: Optional[str],
    dataset_name: str = "dreamerdeo/finqa",
    split: str = "train",
    batch_size: int = 4,
    total_samples: Optional[int] = None,
    output_file: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Pipeline to generate formatted questions using the QuestionGenerator class.

    Args:
        model_name (str): Name of the AI model to use for question generation.
        anthropic_key (str, optional): Anthropic API key if using an Anthropic model.
        dataset_name (str): Name of the dataset to load.
        split (str): Dataset split to use (e.g., 'train', 'test').
        batch_size (int): Number of samples to process in each batch.
        total_samples (int, optional): Total number of samples to process. If None, processes the entire dataset.
        output_file (str, optional): Path to save the generated questions as a JSON or CSV file.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing the generated questions and their analyses.
    """
    # Initialize QuestionGenerator
    question_gen = QuestionGenerator(
        model_name=model_name,
        anthropic_key=anthropic_key
    )
    
    # Load Dataset
    question_gen.load_dataset(dataset_name=dataset_name, split=split)
    
    # Generate Questions
    formatted_questions = question_gen.generate_questions(
        batch_size=batch_size,
        total_samples=total_samples
    )
    
    # Optionally, save the generated questions to a file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_questions, f, ensure_ascii=False, indent=4)
        print(f"Generated questions saved to {output_file}")
    
    return formatted_questions 
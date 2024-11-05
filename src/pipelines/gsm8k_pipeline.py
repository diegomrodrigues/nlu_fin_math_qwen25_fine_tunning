from typing import Optional, Dict, Any, List
import pandas as pd
from deepeval.benchmarks import GSM8K
from handlers.model_handler import ModelHandler
from training.dpo_training import DPOTrainingPipeline
from evaluation.answer_comparator import AnswerComparator
from utils.timers import Timer

def evaluate_gsm8k_pipeline(
    model_path: str,
    n_problems: int = 100,
    n_shots: int = 3,
    enable_cot: bool = True,
    is_dpo_model: bool = False,
    device: str = "cuda",
    max_seq_length: int = 2048,
    mode: str = "cot",
    batch_size: int = 4,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Pipeline to evaluate models using the GSM8K benchmark.

    Args:
        model_path: Path to the model or model name
        n_problems: Number of GSM8K problems to evaluate
        n_shots: Number of few-shot examples to use
        enable_cot: Whether to enable chain-of-thought reasoning
        is_dpo_model: Whether using a DPO-trained model
        device: Device to run evaluation on
        max_seq_length: Maximum sequence length
        mode: Reasoning mode ('cot' or 'tir')
        batch_size: Batch size for processing
        output_file: Optional path to save results

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"Initializing GSM8K evaluation pipeline for {model_path}")
    
    # Initialize GSM8K benchmark
    benchmark = GSM8K(
        n_problems=n_problems,
        n_shots=n_shots,
        enable_cot=enable_cot
    )
    
    # Initialize model based on type
    if is_dpo_model:
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
        
    else:
        model_handler = ModelHandler(
            model_name=model_path,
            device=device
        )
    
    # Create custom model wrapper for GSM8K
    class ModelWrapper:
        def __init__(self, model, is_dpo=False):
            self.model = model
            self.is_dpo = is_dpo
            self.dpo_pipeline = dpo_pipeline if is_dpo else None
            self.model_handler = model_handler if not is_dpo else None
            
        def __call__(self, prompt: str) -> str:
            if self.is_dpo:
                return self.dpo_pipeline.generate_response(
                    model=self.model,
                    tokenizer=tokenizer,
                    question=prompt,
                    mode=mode
                )
            else:
                batch_inputs = self.model_handler.create_batch_inputs(
                    questions=[prompt],
                    batch_size=1,
                    mode=mode
                )
                responses = self.model_handler.generate_responses(batch_inputs)
                return responses[0]["response"]
    
    # Create model wrapper instance
    model_wrapper = ModelWrapper(model, is_dpo=is_dpo_model)
    
    # Run benchmark evaluation
    print("Starting GSM8K evaluation...")
    with Timer("GSM8K Evaluation"):
        benchmark.evaluate(model=model_wrapper)
    
    # Get results
    results = {
        "overall_score": benchmark.overall_score,
        "total_problems": n_problems,
        "correct_answers": int(benchmark.overall_score * n_problems / 100),
        "n_shots": n_shots,
        "enable_cot": enable_cot,
        "model_path": model_path,
        "detailed_results": benchmark.results
    }
    
    # Save results if output file specified
    if output_file:
        results_df = pd.DataFrame(benchmark.results)
        results_df.to_csv(output_file, index=False)
        print(f"Detailed results saved to {output_file}")
    
    print(f"GSM8K Evaluation Results:")
    print(f"Overall Score: {results['overall_score']}%")
    print(f"Correct Answers: {results['correct_answers']}/{results['total_problems']}")
    
    return results 
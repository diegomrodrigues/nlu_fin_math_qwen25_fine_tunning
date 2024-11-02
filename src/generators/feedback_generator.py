from typing import Dict, Any, Optional
import re
import pandas as pd
from tqdm import tqdm
from ..interfaces.anthropic_interface import AnthropicInterface
from ..config.models import MODELS
from ..utils.timers import Timer

FEEDBACK_PROMPT = """
Your task is to improve the AI's response and reasoning to match the reference answer given the question posed to it.

<question>{{question}}</question>
<reference_answer>{{answer}}</reference_answer>

The AI's response was:

<ai_reasoning>
{{cot_response}}
</ai_reasoning>

<ai_answer>
{{extracted_answer}}
</ai_answer>

The reference answer is:

<reference_answer>
{{answer}}
</reference_answer>

Your task:

- Provide a corrected version of the AI's reasoning that fixes the mistakes it made to reach match the <reference_answer> based solo on the <question>. Make sure to include the final answer within \\boxed{}.
- Ensure that the corrected_response is an improved version of the AI's response at <ai_reasoning> without rewriting it entirely. It should preserve most of the AI's reasoning only with minor adjustments to get to the correct answer from the wrong answer provided by the AI on <ai_answer>.

Provide your corrected response, including the final answer within \\boxed{}, based on the <ai_response>, in the following format:

<corrected_response>
[Your corrected ai_reasoning with the final answer within \\boxed{} here]
</corrected_response>
"""

class FeedbackGenerator:
    """Generates feedback using various AI models."""

    def __init__(self, model_name: str, anthropic_key: str):
        """
        Initialize the feedback generator with specified model and API key.

        Args:
            model_name (str): Name of the model to use.
            anthropic_key (str): Anthropic API key.
        """
        self.model_config = MODELS.get(model_name)
        if not self.model_config:
            raise ValueError(f"Invalid model name. Choose from: {list(MODELS.keys())}")

        if not anthropic_key:
            raise ValueError("Anthropic API key required")

        self.model_interface = AnthropicInterface(anthropic_key, self.model_config)
        self.prompt_template = FEEDBACK_PROMPT

    def generate_feedback(
        self,
        wrong_answers_df: pd.DataFrame,
        batch_size: Optional[int] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Generates feedback for wrong answers in batches.

        Args:
            wrong_answers_df: DataFrame containing wrong answers.
            batch_size: Optional batch size for processing.

        Returns:
            Dictionary containing generated feedback.
        """
        if batch_size is None:
            batch_size = self.model_config.batch_size

        total_samples = len(wrong_answers_df)
        formatted_feedback = {}

        for start_idx in tqdm(range(0, total_samples, batch_size)):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_df = wrong_answers_df.iloc[start_idx:end_idx]

            prompts = []
            for _, row in batch_df.iterrows():
                prompt = self.prompt_template
                for key in ["question", "answer", "cot_response", "extracted_answer"]:
                    prompt = prompt.replace(f"{{{{{key}}}}}", str(row[key]))

                prompts.append({
                    "id": str(row.name),  # Using index as ID
                    "PROMPT": prompt,
                    **{
                        key.upper(): str(row[key])
                        for key in ["question", "answer", "cot_response", "extracted_answer"]
                    }
                })

            try:
                custom2prompt = self.model_interface.generate_batches(prompts)

                def extract_corrected_response(text: str) -> str:
                    """Extract the corrected response from model output."""
                    if not text:
                        return ""
                    match = re.search(r"<corrected_response>\s*(.*?)\s*</corrected_response>",
                                    text, re.DOTALL | re.IGNORECASE)
                    return match.group(1).strip() if match else ""

                for response in self.model_interface.retrieve_batches_results():
                    try:
                        model_output = response["MODEL_OUTPUT"]
                        prompt_data = custom2prompt[response["id"]]

                        corrected_response = extract_corrected_response(model_output)

                        if corrected_response:
                            formatted_feedback[prompt_data["id"]] = {
                                "CORRECTED_RESPONSE": corrected_response,
                                **prompt_data
                            }
                        else:
                            print(f"Missing corrected response for {response['id']}")

                    except Exception as e:
                        print(f"Error processing response {response['id']}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing batch {start_idx}-{end_idx}: {str(e)}")
                continue

        return formatted_feedback

    def save_feedback_to_csv(self, feedback_data: Dict[str, Dict[str, str]], output_file: str):
        """
        Saves generated feedback to a CSV file.

        Args:
            feedback_data: Dictionary containing feedback data.
            output_file: Path to output CSV file.
        """
        feedback_df = pd.DataFrame.from_dict(feedback_data, orient='index')
        feedback_df.to_csv(output_file, index=True)
        print(f"Feedback saved to {output_file}") 
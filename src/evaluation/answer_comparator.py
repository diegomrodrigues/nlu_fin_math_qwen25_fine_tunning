import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import pandas as pd
from ..utils.timers import Timer

class AnswerComparator:
    """Compares AI-generated answers against correct answers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the AnswerComparator with Qwen model and configurations.

        Args:
            model_name: Name or path of the Qwen model.
            device: Device to run the model on ("cuda" or "cpu").
            system_prompt: Custom system prompt. If None, uses default prompt.
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Default system prompt if none provided
        self.system_prompt = system_prompt if system_prompt else (
            "You are a math teacher evaluating student answers. "
            "Your task is to determine if the student's answer matches the correct answer. "
            "Respond with 'Correct' if the answer matches, and 'Incorrect' if it doesn't."
        )

        # Precompute token IDs for 'Correct' and 'Wrong'
        self.correct_token_id = self.tokenizer.encode("Correct", add_special_tokens=False)[0]
        self.wrong_token_id = self.tokenizer.encode("Wrong", add_special_tokens=False)[0]

        # Setup logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.ERROR)

    def build_messages(
        self,
        question: str,
        student_answer: str,
        correct_answer: str
    ) -> List[Dict[str, str]]:
        """
        Build message list for the Qwen chat template.

        Args:
            question: The question posed to the student.
            student_answer: The student's answer.
            correct_answer: The correct answer.

        Returns:
            List of message dictionaries.
        """
        evaluation_prompt = f"""
Question:
{question}

Student's Answer:
{student_answer}

Correct Answer:
{correct_answer}

Is the student's final answer correct? Respond with either 'Correct' or 'Wrong'.
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": evaluation_prompt}
        ]
        return messages

    def get_model_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Get response from the Qwen model for given messages.

        Args:
            messages: List of message dictionaries.

        Returns:
            Model response as a string.
        """
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32,  # Shorter for simple correct/incorrect responses
                temperature=0,  # Deterministic output
                do_sample=False
            )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        except Exception as e:
            self.logger.error(f"Error generating model response: {str(e)}")
            return ""

    def get_model_response_with_logprobs(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[str, float]:
        """
        Get response from the Qwen model along with confidence score based on logprobs.

        Args:
            messages: List of message dictionaries.

        Returns:
            Tuple of (response_text, confidence_score).
        """
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=32,
                    temperature=0,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            generated_ids = [
                output_ids[len(model_inputs.input_ids[0]):]
                for output_ids in outputs.sequences
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Calculate confidence score from logits of the last relevant token
            last_token_scores = outputs.scores[-1][0]  # Shape: [vocab_size]

            token_probs = F.softmax(last_token_scores, dim=0)

            correct_prob = token_probs[self.correct_token_id].item()
            wrong_prob = token_probs[self.wrong_token_id].item()

            confidence = correct_prob / (correct_prob + wrong_prob)

            return response, confidence

        except Exception as e:
            self.logger.error(f"Error generating model response: {str(e)}")
            return "", 0.5

    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """
        Extracts the answer from within \boxed{} notation.

        Args:
            text: The text containing the boxed answer.

        Returns:
            The extracted answer as a string, or None if not found.
        """
        if not isinstance(text, str):
            return None

        pattern = r'\\boxed\{([^{}]+)\}'
        matches = re.findall(pattern, text)

        if not matches:
            return None

        return matches[-1].strip()

    def process_dataframe(
        self,
        df: pd.DataFrame,
        question_col: str,
        response_col: str,
        answer_col: str,
        with_confidence: bool = False
    ) -> pd.DataFrame:
        """
        Processes a dataframe and adds comparison metrics.

        Args:
            df: Input dataframe.
            question_col: Column name containing questions.
            response_col: Column name containing student responses.
            answer_col: Column name containing correct answers.
            with_confidence: Whether to include confidence scores.

        Returns:
            DataFrame with added comparison metrics.
        """
        result_df = df.copy()

        result_df['model_evaluation'] = ''
        result_df['is_correct'] = False
        if with_confidence:
            result_df['confidence_score'] = 0.0

        for idx, row in result_df.iterrows():
            try:
                extracted_answer = self.extract_boxed_answer(row[response_col])

                messages = self.build_messages(
                    row[question_col],
                    extracted_answer,
                    row[answer_col]
                )

                if not with_confidence:
                    model_eval = self.get_model_response(messages)
                    confidence = 0.0
                else:
                    model_eval, confidence = self.get_model_response_with_logprobs(messages)

                result_df.at[idx, 'extracted_answer'] = extracted_answer
                result_df.at[idx, 'model_evaluation'] = model_eval
                result_df.at[idx, 'is_correct'] = self._evaluate_correctness(model_eval)

                if with_confidence:
                    result_df.at[idx, 'confidence_score'] = confidence

            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {str(e)}")
                continue

        return result_df

    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate overall metrics for the processed dataframe.

        Args:
            df: Processed dataframe.

        Returns:
            Dictionary of calculated metrics.
        """
        try:
            metrics = {
                'accuracy': df['is_correct'].mean(),
                'total_samples': len(df),
                'correct_samples': df['is_correct'].sum(),
                'average_confidence': df['confidence_score'].mean() if 'confidence_score' in df.columns else None
            }
            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _evaluate_correctness(self, model_eval: str) -> bool:
        """
        Evaluate if the model's response indicates a correct answer.

        Args:
            model_eval: The model's evaluation response.

        Returns:
            True if correct, False otherwise.
        """
        return 'correct' in model_eval.lower() 
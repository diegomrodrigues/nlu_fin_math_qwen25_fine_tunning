# -*- coding: utf-8 -*-
"""Como aproveitar ao máximo sua assinatura do Colab

Automatically consolidated and optimized version of the original Colab notebook.

### TIR Integration
"""

# ## Install Dependencies
# Install all required packages in a single cell to optimize runtime and reduce redundancy.

!pip install -q datasets trl unsloth anthropic google-generativeai pint

# ## Import Statements
# Organized all imports at the top and removed duplicates.

import re
import os
import sys
import time
import torch
import logging
import pandas as pd
import numpy as np
from io import StringIO
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from contextlib import contextmanager

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from tqdm import tqdm
import google.generativeai as genai
import anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

# ## Utility Classes

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        print(f'{self.name} took {self.duration:.2f} seconds')

class OutputCapture:
    """Manages the capture of stdout during code execution."""
    
    @staticmethod
    @contextmanager
    def capture_stdout():
        stdout = StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout
        try:
            yield stdout
        finally:
            sys.stdout = old_stdout

    @staticmethod
    def execute_code(code_block: str) -> str:
        """Executes a block of code and returns its output."""
        with Timer("Code execution"):
            with OutputCapture.capture_stdout() as output:
                try:
                    local_vars = {}
                    exec(code_block.strip(), {}, local_vars)
                    return output.getvalue().strip()
                except Exception as e:
                    print(f"Code execution error: {str(e)}")
                    return f"Error: {str(e)}"

class CodeBlockStoppingCriteria(StoppingCriteria):
    """Stopping criteria for code blocks."""
    def __init__(self, tokenizer, device):
        super().__init__()
        self.python_start_pattern = "```output"
        self.python_start_tokens = tokenizer(
            [self.python_start_pattern],
            return_tensors="pt"
        ).to(device)
        self.start_token_ids = self.python_start_tokens.input_ids[0]
        self.pattern_length = len(self.start_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        last_tokens = input_ids[:, -self.pattern_length:]
        matches = torch.all(last_tokens == self.start_token_ids, dim=1)
        return torch.any(matches).item()

# ## Model Handler

class ModelHandler:
    """Manages AI model operations."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"Initializing ModelHandler with model: {model_name}")
        self.model, self.tokenizer = self._setup_model_and_tokenizer()
        self.stopping_criteria = StoppingCriteriaList([
            CodeBlockStoppingCriteria(self.tokenizer, self.model.device)
        ])

    def _setup_model_and_tokenizer(self):
        """Initialize model with optimizations."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for faster inference
            device_map="auto",
            use_cache=True
        )
        model.config.use_cache = True

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            model_max_length=2048  # Set explicit max length
        )
        tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

        return model, tokenizer

    def create_batch_inputs(
        self,
        questions: List[str],
        ids: Optional[List[str]] = None,
        batch_size: int = 8,
        mode: str = "tir"
    ) -> List[Dict]:
        """Creates batch inputs for the model."""
        if ids is None:
            ids = [f"question-{i}" for i in range(len(questions))]

        with Timer("Batch input creation"):
            questions = [str(q) for q in questions]
            all_inputs = []

            system_message = (
                "Please integrate natural language reasoning with programs to solve the problem above, "
                "and put your final answer within \\boxed{}."
                if mode == "tir"
                else "Please reason step by step, and put your final answer within \\boxed{}."
            )

            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                print(f"Processing batch {i//batch_size + 1} with {len(batch_questions)} questions")

                batch_messages = [
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": q}
                    ]
                    for q in batch_questions
                ]

                batch_texts = self.tokenizer.apply_chat_template(
                    batch_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    return_tensors=True
                )

                model_inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_attention_mask=True
                ).to(self.model.device)

                all_inputs.append({
                    "ids": batch_ids,
                    "inputs": model_inputs,
                    "questions": batch_questions
                })

        return all_inputs

    @torch.no_grad()
    def generate_responses(self, batched_inputs: List[Dict]) -> List[Dict[str, Any]]:
        """Generates responses for batched inputs."""
        all_responses = []

        for batch_idx, batch in enumerate(batched_inputs):
            try:
                print(f"Starting generation for batch {batch_idx + 1}")

                with Timer(f"Model generation for batch {batch_idx + 1}"):
                    batch_inputs = batch["inputs"]

                    generated_ids = self.model.generate(
                        **batch_inputs,
                        max_new_tokens=512,
                        stopping_criteria=self.stopping_criteria,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        num_return_sequences=1,
                        use_cache=True,
                    )

                with Timer("Code execution and generation continuation"):
                    generated_ids = self._execute_and_continue_generation(generated_ids)

                with Timer("Response decoding"):
                    generated_ids = [
                        output_ids[len(input_ids):]
                        for input_ids, output_ids in zip(batch_inputs.input_ids, generated_ids)
                    ]

                    responses = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False  # Faster decoding
                    )

                for i, response in enumerate(responses):
                    question = batch["questions"][i]
                    question_id = batch["ids"][i]

                    all_responses.append({
                        "id": question_id,
                        "question": question,
                        "response": response
                    })

                print(f"Successfully processed batch {batch_idx + 1}")

            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {str(e)}")

        return all_responses

    def _execute_and_continue_generation(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Executes generated code and continues generation."""
        with Timer("Code execution and continuation"):
            current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            code_match = re.search(r'```python\n(.*?)\n```', current_text, re.DOTALL)

            if not code_match:
                return input_ids

            code_block = code_match.group(1)
            execution_result = OutputCapture.execute_code(code_block)
            continuation = f"\n{execution_result}\n```"

            with Timer("Continuation tokenization"):
                continuation_tokens = self.tokenizer(
                    continuation,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids.to(self.model.device)

            new_input_ids = torch.cat([input_ids, continuation_tokens], dim=1)

            with Timer("Final generation"):
                outputs = self.model.generate(
                    input_ids=new_input_ids,
                    max_new_tokens=64,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            return outputs

# ## DPO Training Pipeline

@dataclass
class ModelConfig:
    """Base configuration for AI models."""
    name: str
    max_tokens: int = 1024
    temperature: float = 0.7
    batch_size: int = 1000
    provider: 'ModelProvider' = 'ModelProvider.ANTHROPIC'

class ModelProvider(Enum):
    """Supported AI model providers."""
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

class DPOTrainingPipeline:
    """Pipeline for Direct Preference Optimization training with enhanced prompt formatting."""

    def __init__(
        self,
        model_name: str = "unsloth/Qwen2.5-Math-1.5B-bnb-4bit",
        original_model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
        max_seq_length: int = 2048,
        output_dir: str = "./dpo_output",
        beta: float = 0.1,
        batch_size: int = 4,
        num_train_epochs: int = 3,
        learning_rate: float = 5e-5,
        mode: str = "cot"
    ):
        self.model_name = model_name
        self.original_model_name = original_model_name
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.beta = beta
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.mode = mode

        print(f"Initializing DPO Training Pipeline with model: {model_name}")
        print(f"Using reasoning mode: {mode}")

    def format_prompt_for_dpo(self, question: str, mode: str = "cot") -> str:
        """
        Format a question into a prompt suitable for DPO training.
        Returns a single string instead of a list of messages.
        """
        system_message = {
            "tir": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.",
            "cot": "Please reason step by step, and put your final answer within \\boxed{}."
        }

        chat_messages = [
            {"role": "system", "content": system_message[mode]},
            {"role": "user", "content": str(question)}
        ]

        if not hasattr(self, 'tokenizer'):
            _, _, self.tokenizer = self.prepare_model_and_tokenizer()

        formatted_prompt = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors=False  # Return as strings
        )

        return formatted_prompt

    def prepare_model_and_tokenizer(self) -> Tuple[torch.nn.Module, Any, Any]:
        """Initialize the model and tokenizer with optimizations."""
        print("Loading model and tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            self.original_model_name,
            padding_side="left",
            model_max_length=self.max_seq_length  # Set explicit max length
        )
        tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

        model, fast_tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect dtype
            load_in_4bit=True,
        )

        # Add LoRA weights with optimized configuration
        model = FastLanguageModel.get_peft_model(
            model,
            r=4,
            target_modules=[
                "up_proj", "down_proj"
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=1337,
        )

        return model, fast_tokenizer, tokenizer

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        question_col: str = "QUESTION",
        chosen_col: str = "CORRECTED_RESPONSE",
        rejected_col: str = "AI_RESPONSE"
    ) -> Dataset:
        """
        Prepare the dataset for DPO training with formatted prompts.
        """
        print("Preparing training dataset...")

        required_cols = [question_col, chosen_col, rejected_col]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Required: {required_cols}")

        if not hasattr(self, 'tokenizer'):
            _, _, self.tokenizer = self.prepare_model_and_tokenizer()

        system_message = {
            "tir": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.",
            "cot": "Please reason step by step, and put your final answer within \\boxed{}."
        }[self.mode]

        chat_messages = [
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": str(question)}
            ]
            for question in df[question_col]
        ]

        formatted_prompts = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors=False  # Return as strings
        )

        train_dataset_dict = {
            "prompt": formatted_prompts,
            "chosen": df[chosen_col].astype(str).tolist(),
            "rejected": df[rejected_col].astype(str).tolist()
        }

        train_dataset = Dataset.from_dict(train_dataset_dict)

        print(f"Created dataset with {len(train_dataset)} examples")
        return train_dataset

    def create_training_config(self) -> DPOConfig:
        """Create the DPO training configuration."""
        print("Creating training configuration...")

        training_args = DPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.learning_rate,
            beta=self.beta,
            max_length=self.max_seq_length,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            save_strategy="epoch",
            evaluation_strategy="no",
            logging_steps=10,
            optim="paged_adamw_32bit",
            fp16=True,
        )

        return training_args

    def train(self, train_dataset: Dataset) -> None:
        """Execute the DPO training process."""
        print("Starting DPO training...")

        model, tokenizer, _ = self.prepare_model_and_tokenizer()
        training_args = self.create_training_config()

        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # No reference model needed
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )

        print("Beginning training loop...")
        trainer.train()

        print(f"Saving model to {self.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)

    @classmethod
    def load_from_pretrained(
        cls,
        model_path: str,
        device: str = "auto",
        max_seq_length: int = 2048,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True
    ) -> Tuple[torch.nn.Module, Any]:
        """
        Load a fine-tuned model and tokenizer from a directory.
        """
        print(f"Loading fine-tuned model from {model_path}")

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit
            )
            print("Model and tokenizer loaded successfully")
            return model, tokenizer

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def generate_response(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        question: str,
        mode: str = "cot",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response using the loaded model.
        """
        prompt = self.format_prompt_for_dpo(question, mode=mode)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
            padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

# ## Question Generation

PROMPT = """
You are an expert financial analyst tasked with creating comprehensive, self-contained questions based on complex financial reports. Your goal is to reformulate given information into a single question that includes all necessary context for answering without additional information.

Here is the financial report you need to analyze:

<pre_text>
{{PRE_TEXT}}
</pre_text>

<table>
{{TABLE}}
</table>

<post_text>
{{POST_TEXT}}
</post_text>

For reference, here is the original question and its answer:

<question>{{QUESTION}}</question>
<answer>{{ANSWER}}</answer>

And here is the key evidence used to answer the original question:

<gold_evidence>{{GOLD_EVIDENCE}}</gold_evidence>

Your task is to create a new, more comprehensive question that incorporates all the necessary information from the provided text, including the details from the gold evidence. Follow these steps:

1. Carefully analyze the given text, extracting and listing key financial figures, percentages, dates, and any other relevant information.
2. Identify the main topic or focus of the original question and summarize it.
3. Compare the original question with the gold evidence to ensure all crucial information is captured.
4. Extract all relevant information from the provided text that is necessary to answer the question.
5. Incorporate this information into a new, single question that is self-contained and provides all the context needed to solve it.
6. Ensure that the new question is clear, concise, and specific.
7. Include all necessary numerical values, dates, or other details within the question itself.

Guidelines for the new question:
- It should be possible to answer the question using only the information provided within the question text.
- Avoid references to external documents or additional context.
- Use clear and precise language.
- Make sure to create a single question, not a list of questions.
- The question should provide all data necessary to answer it within its text.

Before formulating your final question, wrap your analysis in <analysis> tags to break down the information and show your thought process. This will help ensure a thorough interpretation of the data.

Format your output as follows:

<analysis>
[Your detailed analysis of the report, breaking down key facts and their relationships]
</analysis>

<new_question>
[Your newly formulated, auto-explainable question with all necessary details from the analysis in order to answer the question]
</new_question>

Ensure that your new question is sufficiently detailed and self-contained so that it can be answered accurately without any additional context.
"""

class AnthropicInterface:
    """Interface for interacting with Anthropic models."""

    def __init__(self, api_key: str, model_config: ModelConfig):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_config = model_config
        self.batch_ids = []

    def generate_batches(self, prompts: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """Generates batches for processing."""
        try:
            custom2prompt = {
                f"batch-example-{i}": prompt_data for i, prompt_data in enumerate(prompts)
            }
            id2custom = {
                prompt_data["id"]: f"batch-example-{i}" for i, prompt_data in enumerate(prompts)
            }

            requests = [
                Request(
                    custom_id=id2custom[prompt_data["id"]],
                    params=MessageCreateParamsNonStreaming(
                        model=self.model_config.name,
                        max_tokens=self.model_config.max_tokens,
                        temperature=self.model_config.temperature,
                        messages=[
                            {"role": "user", "content": prompt_data["PROMPT"]}
                        ]
                    )
                )
                for prompt_data in prompts
            ]

            batch_response = self.client.beta.messages.batches.create(requests=requests)
            self.batch_ids.append(batch_response.id)

            return custom2prompt

        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            raise e

    def _wait_for_batch_completion(self, batch_id: str, poll_interval: int = 5) -> bool:
        """
        Waits for a batch to complete by polling its status.

        Args:
            batch_id: The ID of the batch to check.
            poll_interval: Time in seconds between status checks.

        Returns:
            bool: True if batch completed successfully, False otherwise.
        """
        while True:
            try:
                batch_status = self.client.beta.messages.batches.retrieve(
                    batch_id,
                    extra_headers={"anthropic-beta": "message-batches-2024-09-24"}
                )

                print(f"Checking batch: {batch_id} {batch_status.processing_status}")

                if batch_status.processing_status == "ended":
                    if batch_status.request_counts.succeeded == (batch_status.request_counts.processing + batch_status.request_counts.succeeded):
                        return True
                    else:
                        print(f"Batch {batch_id} completed with errors:")
                        print(f"Succeeded: {batch_status.request_counts.succeeded}")
                        print(f"Errored: {batch_status.request_counts.errored}")
                        print(f"Canceled: {batch_status.request_counts.canceled}")
                        print(f"Expired: {batch_status.request_counts.expired}")
                        return False

                time.sleep(poll_interval)

            except Exception as e:
                print(f"Error checking batch {batch_id} status: {str(e)}")
                return False

    def retrieve_batches_results(self):
        """Generator that yields results from completed batch requests."""
        for batch_id in self.batch_ids:
            try:
                if self._wait_for_batch_completion(batch_id):
                    for result in self.client.beta.messages.batches.results(batch_id):
                        try:
                            response_content = result.result.message.content[0].text
                            yield {
                                "id": result.custom_id,
                                "MODEL_OUTPUT": response_content
                            }
                        except Exception as e:
                            print(f"Error processing result: {str(e)}")
                            continue
                else:
                    print(f"Skipping batch {batch_id} due to completion failure")

            except Exception as e:
                print(f"Error retrieving batch {batch_id}: {str(e)}")
                continue

class QuestionGenerator:
    """Generates questions using various AI models."""

    def __init__(self, model_name: str, anthropic_key: Optional[str] = None, gemini_key: Optional[str] = None):
        """
        Initialize the question generator with specified model and API keys.

        Args:
            model_name (str): Name of the model to use.
            anthropic_key (str, optional): Anthropic API key.
            gemini_key (str, optional): Google API key.
        """
        self.model_config = MODELS.get(model_name)
        if not self.model_config:
            raise ValueError(f"Invalid model name. Choose from: {list(MODELS.keys())}")

        if self.model_config.provider == ModelProvider.ANTHROPIC:
            if not anthropic_key:
                raise ValueError("Anthropic API key required for Claude models")
            self.model_interface = AnthropicInterface(anthropic_key, self.model_config)

        self.prompt_template = PROMPT  # Your prompt template here

    def load_dataset(self, dataset_name: str = "dreamerdeo/finqa", split: str = "train") -> None:
        """Load and preprocess the dataset."""
        print(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name)
        self.train_dataset = self.dataset[split]

    def generate_questions(
        self,
        batch_size: Optional[int] = None,
        total_samples: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generates formatted questions.

        Args:
            batch_size: Batch size for processing.
            total_samples: Total number of samples to process.

        Returns:
            Dictionary containing formatted questions.
        """
        if batch_size is None:
            batch_size = self.model_config.batch_size

        if not total_samples:
            total_samples = len(self.train_dataset)

        formatted_questions = {}

        for start_idx in tqdm(range(0, total_samples, batch_size)):
            end_idx = min(start_idx + batch_size, total_samples)

            prompts = []
            for i in range(start_idx, end_idx):
                prompt = self.prompt_template
                for k, v in self.train_dataset[i].items():
                    prompt = prompt.replace(f"{{{{{k.upper()}}}}}", str(v))

                prompts.append({
                    "id": self.train_dataset[i]["id"],
                    "PROMPT": prompt,
                    **{
                        field.upper(): str(self.train_dataset[i][field])
                        for field in ["pre_text", "table", "post_text", "question", "gold_evidence", "answer"]
                    }
                })

            try:
                custom2prompt = self.model_interface.generate_batches(prompts)

                def clean_extracted_text(text: str) -> str:
                    if not text:
                        return ""
                    text = re.sub(r'\n\s*\n', '\n', text)
                    text = '\n'.join(line.strip() for line in text.split('\n'))
                    return text.strip()

                for response in self.model_interface.retrieve_batches_results():
                    try:
                        model_output = response["MODEL_OUTPUT"]
                        prompt_data = custom2prompt[response["id"]]

                        analysis_match = re.search(r"<analysis>\s*(.*?)\s*</analysis>",
                                                 model_output, re.DOTALL | re.IGNORECASE)
                        question_match = re.search(r"<new_question>\s*(.*?)\s*</new_question>",
                                                 model_output, re.DOTALL | re.IGNORECASE)

                        analysis = clean_extracted_text(analysis_match.group(1)) if analysis_match else ""
                        question = clean_extracted_text(question_match.group(1)) if question_match else ""

                        formatted_questions[prompt_data["id"]] = {
                            "ANALYSIS": analysis,
                            "FMT_QUESTION": question,
                            **prompt_data
                        }

                        if not analysis or not question:
                            print(f"Missing {'analysis' if not analysis else 'question'} for {response['id']}")

                    except Exception as e:
                        print(f"Error processing response {response['id']}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing batch {start_idx}-{end_idx}: {str(e)}")
                continue

        return formatted_questions

# ## Feedback Generation

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

# ## Evaluation

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

# ## Model Configurations

# Define available models configurations
MODELS = {
    # Anthropic models
    "claude-sonnet": ModelConfig(
        name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        max_tokens=4096,
        temperature=0
    ),
    "claude-haiku": ModelConfig(
        name="claude-3-haiku-20240307",
        provider=ModelProvider.ANTHROPIC,
        max_tokens=4096,
        temperature=0
    ),
    "claude-opus": ModelConfig(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC
    )
}

# ## Execution Steps

# ### Question Generation Pipeline

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
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_questions, f, ensure_ascii=False, indent=4)
        print(f"Generated questions saved to {output_file}")
    
    return formatted_questions

# ### Feedback Generation Pipeline

def generate_feedback_pipeline(
    model_name: str,
    anthropic_key: str,
    wrong_answers_df: pd.DataFrame,
    batch_size: int = 5,
    output_file: Optional[str] = None
) -> Dict[str, Dict[str, str]]:
    """
    Pipeline to generate feedback for wrong answers using the FeedbackGenerator class.

    Args:
        model_name (str): Name of the AI model to use for feedback generation.
        anthropic_key (str): Anthropic API key.
        wrong_answers_df (pd.DataFrame): DataFrame containing wrong answers with required columns.
        batch_size (int): Number of samples to process in each batch.
        output_file (str, optional): Path to save the generated feedback as a CSV file.

    Returns:
        Dict[str, Dict[str, str]]: Dictionary containing the generated feedback.
    """
    # Initialize FeedbackGenerator
    feedback_gen = FeedbackGenerator(
        model_name=model_name,
        anthropic_key=anthropic_key
    )
    
    # Generate Feedback
    feedback_data = feedback_gen.generate_feedback(
        wrong_answers_df=wrong_answers_df,
        batch_size=batch_size
    )
    
    # Optionally, save the feedback to a CSV file
    if output_file:
        feedback_gen.save_feedback_to_csv(feedback_data, output_file)
    
    return feedback_data

# ### DPO Training Pipeline

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

# ### Example Usage of the Pipelines

# ```python
# # Example usage for generating questions
# generated_questions = generate_questions_pipeline(
#     model_name="your-model-name",
#     anthropic_key="your-anthropic-api-key",
#     dataset_name="your-dataset-name",
#     split="train",
#     batch_size=8,
#     total_samples=100,
#     output_file="generated_questions.json"
# )

# # Example usage for generating feedback
# feedback = generate_feedback_pipeline(
#     model_name="claude-sonnet",
#     anthropic_key="your-anthropic-api-key",
#     wrong_answers_df=wrong_answers,  # DataFrame containing wrong answers
#     batch_size=5,
#     output_file="feedback.csv"
# )

# # Example usage for training via DPO
# train_dpo_pipeline(
#     model_name="unsloth/Qwen2.5-Math-1.5B-bnb-4bit",
#     original_model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
#     max_seq_length=2048,
#     output_dir="./dpo_output",
#     beta=0.1,
#     batch_size=4,
#     num_train_epochs=3,
#     learning_rate=5e-5,
#     mode="cot",
#     training_data_df=training_data_df,  # DataFrame containing training data
#     question_col="QUESTION",
#     chosen_col="CORRECTED_RESPONSE",
#     rejected_col="AI_RESPONSE"
# )
# ```

# ### Notes:
# - Replace `"your-model-name"`, `"your-anthropic-api-key"`, and other placeholder strings with your actual model names and API keys.
# - Ensure that the `training_data_df` provided to the `train_dpo_pipeline` contains the required columns: `QUESTION`, `CORRECTED_RESPONSE`, and `AI_RESPONSE`.
# - The example usage section is commented out. Uncomment and modify it as needed to run the pipelines.

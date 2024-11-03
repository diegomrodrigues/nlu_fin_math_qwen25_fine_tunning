from dataclasses import dataclass, field
from typing import Tuple, List, Any
import pandas as pd
from transformers import AutoTokenizer
from config.models import MODELS
from handlers.model_handler import ModelHandler
from utils.timers import Timer
from utils.output_capture import OutputCapture
from interfaces.anthropic_interface import AnthropicInterface
from unsloth import FastLanguageModel # type: ignore
from trl import DPOTrainer, DPOConfig # type: ignore
from datasets import Dataset
import torch # type: ignore
from peft import AutoPeftModelForCausalLM # type: ignore
import os
from huggingface_hub import HfApi
import wandb # type: ignore

@dataclass
class DPOTrainingPipeline:
    """Pipeline for Direct Preference Optimization training with enhanced prompt formatting."""
    project_name: str
    model_name: str = "unsloth/Qwen2.5-Math-1.5B-bnb-4bit"
    original_model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    max_seq_length: int = 2048
    output_dir: str = "./dpo_output"
    beta: float = 0.1
    batch_size: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    mode: str = "cot"
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0
    lora_target_modules: List[str] = field(default_factory=lambda: ["up_proj", "down_proj"])
    use_gradient_checkpointing: bool = True
    fp16: bool = True
    optim: str = "paged_adamw_32bit"
    save_strategy: str = "epoch"
    random_state: int = 1337

    def __post_init__(self):
        print(f"Initializing DPO Training Pipeline with model: {self.model_name}")
        print(f"Project name: {self.project_name}")
        print(f"Using reasoning mode: {self.mode}")

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

    def prepare_model_and_tokenizer(self) -> Tuple[torch.nn.Module, Any, Any]: # type: ignore
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
            r=self.lora_r,
            target_modules=self.lora_target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            random_state=self.random_state,
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
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            beta=self.beta,
            max_length=self.max_seq_length,
            gradient_checkpointing=self.use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            lr_scheduler_type="cosine",
            warmup_ratio=self.warmup_ratio,
            save_strategy=self.save_strategy,
            evaluation_strategy="no",
            logging_steps=5,
            optim=self.optim,
            fp16=self.fp16,
            report_to="wandb",
        )

        return training_args

    def push_to_hub(self, model, tokenizer):
        """Push the model and tokenizer to HuggingFace Hub"""
        try:
            api = HfApi()
            repo_id = f"{api.whoami()['name']}/{self.project_name}"
            
            print(f"Pushing model and tokenizer to HuggingFace Hub: {repo_id}")
            
            # Create repo if it doesn't exist
            api.create_repo(repo_id, exist_ok=True)
            
            # Push model and tokenizer
            model.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            
            print("Successfully pushed to HuggingFace Hub")
            
        except Exception as e:
            print(f"Error pushing to HuggingFace Hub: {str(e)}")
            raise

    def train(self, train_dataset: Dataset) -> None:
        """Execute the DPO training process."""
        print("Starting DPO training...")

        model, tokenizer, _ = self.prepare_model_and_tokenizer()
        training_args = self.create_training_config()

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )

        print("Beginning training loop...")
        trainer.train()

        print(f"Saving model to {self.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)

        # Push to HuggingFace Hub
        self.push_to_hub(model, tokenizer)

        # Close wandb run
        wandb.finish()

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
        Load a fine-tuned model and tokenizer from a directory, handling both base and LoRA models.
        """
        print(f"Loading fine-tuned model from {model_path}")
        
        try:
            # Check if this is a LoRA adapter
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                print("Detected LoRA adapter, loading with PEFT...")
                # First load the base model
                base_model_name = cls.original_model_name  # You might want to store this in adapter_config
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model_name,
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit
                )
                
                # Then load the LoRA adapter
                model = AutoPeftModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto" if device == "auto" else device,
                    torch_dtype=torch.float16
                )
            else:
                print("Loading as complete model...")
                # Load as a complete model
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
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
    ) -> str:
        """
        Generate a deterministic response using the loaded model.
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.random_state)
        
        prompt = self.format_prompt_for_dpo(question, mode=mode)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Use greedy decoding instead of sampling
                num_beams=1  # Use simple greedy search
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
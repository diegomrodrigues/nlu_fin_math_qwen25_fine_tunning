from typing import List, Dict, Any, Optional
import re
import torch # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList
)
from utils.timers import Timer
from utils.output_capture import OutputCapture
from utils.stopping_criteria import CodeBlockStoppingCriteria

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
from typing import List, Optional
import torch
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass

@dataclass
class RewardModelHandler:
    """Handles reward model scoring for best-of-n sampling."""
    model_name: str = "Qwen/Qwen2.5-Math-RM-72B"
    device: str = "cuda"
    
    def __post_init__(self):
        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
    
    def score_responses(
        self,
        questions: List[str],
        responses: List[str],
        system_prompt: Optional[str] = None
    ) -> List[float]:
        """Score a batch of responses using the reward model."""
        if system_prompt is None:
            system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
            
        scores = []
        for question, response in zip(questions, responses):
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ]
            
            conversation_str = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False
            )
            
            input_ids = self.tokenizer.encode(
                conversation_str,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                score = outputs[0].mean().item()
                scores.append(score)
                
        return scores 
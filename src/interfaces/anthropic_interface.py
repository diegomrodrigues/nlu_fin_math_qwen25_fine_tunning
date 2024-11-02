import time
import re
from typing import List, Dict, Any
import anthropic
import pandas as pd
from trl import DPOTrainer, DPOConfig
from dataclasses import dataclass
from ..config.models import ModelConfig, ModelProvider
from ..utils.timers import Timer

@dataclass
class AnthropicInterface:
    """Interface for interacting with Anthropic models."""
    api_key: str
    model_config: ModelConfig
    client: Any = None
    batch_ids: List[str] = None

    def __post_init__(self):
        self.client = anthropic.Anthropic(api_key=self.api_key)
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
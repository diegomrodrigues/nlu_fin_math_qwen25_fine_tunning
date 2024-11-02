from typing import List, Dict, Any, Optional
import re
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from config.models import MODELS
from interfaces.anthropic_interface import AnthropicInterface
from utils.timers import Timer

QUESTION_PROMPT = """
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

        self.prompt_template = QUESTION_PROMPT

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
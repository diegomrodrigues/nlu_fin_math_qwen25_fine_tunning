# AI Model Project

## Overview
A project structured to handle AI model operations, including question generation, feedback generation, and model training using Direct Preference Optimization (DPO).

## Project Structure
- `src/`: Contains all source code divided into modules.
  - `config/`: Model configurations.
  - `utils/`: Utility classes and functions.
  - `handlers/`: Classes handling model operations.
  - `training/`: DPO training pipeline.
  - `interfaces/`: Interfaces for interacting with external APIs.
  - `generators/`: Classes for generating questions and feedback.
  - `evaluation/`: Classes for evaluating AI-generated answers.
  - `pipelines/`: Modular pipelines for various tasks.
- `main.py`: Entry point demonstrating pipeline usage.
- `requirements.txt`: Project dependencies.
- `README.md`: Project documentation.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   - Replace `"your-anthropic-api-key"` in `main.py` with your actual Anthropic API key.
   - Ensure other necessary API keys are set where required.

4. **Prepare Data:**
   - Ensure `wrong_answers.csv` and `training_data.csv` are available in the root directory with the required columns.

## Usage

Run the main script to execute the pipelines:

# LLM Bias Code
* Primary code is in `build_experiments.py`
* Base prompts are in `inputs/writing_samples`
* GPT-suggested substitutions are in `experiments/substitutions.json`
* Experiments to be run are in `experiments/experiments.csv`

## Experiment Prompts & Results

### Experiment Directories
* `experiments/test_mini/` - gpt4-mini, GPT-generated substitutions, "grade-only" prompting
* `experiments/test1/` - gpt4, GPT-generated substitutions, "grade-only" prompting
* `experiments/test2/` - gpt4-mini, manually adjusted substitutions, "grade-only" prompting
* `experiments/test2_textual/` - gpt4-mini, manually adjusted substitutions, "feedback + grade" prompting

### Files
* `substitutions.json` - the substitutions used for each experiment
* `generated_samples.csv` - the generated samples for each experiment, containing the substituted text and metadata for each sample
* `experiments.csv` - List of all experimental runs with metadata
* `experiments_with_responses.csv` - List of all experimental runs with metadata and model response
* `logs.json` - List of all API calls made, with metadata and model response
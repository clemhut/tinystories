# DPNLP Project: Reproducing test results from the TinyStories Paper

Created by Alexander Grabner, Clemens Huetter and Brandon Nader

# Repo Description
## Creating the conda env
```bash
conda env create -f environment.yml
conda activate chatbot
```


## Running the smoke tests
Run the smoke-test module from the repository root:

```bash
python -m src.tests
```

## DPNLP pipeline
The reproduction pipeline lives in `src/dpnlp_pipeline`.

Inspect the configured experiment setup:

```bash
python -m src.dpnlp_pipeline describe
```

Write the default GPT-Eval prompt set to the artifacts directory:

```bash
python -m src.dpnlp_pipeline write-prompts
```

Train one model or all five README-sized models:

```bash
python -m src.dpnlp_pipeline train --model-size xs
python -m src.dpnlp_pipeline train --model-size small
python -m src.dpnlp_pipeline train-all
```

Generate from a trained checkpoint:

```bash
python -m src.dpnlp_pipeline generate --model-size small --prompt "Once upon a time, there was "
```

Generate all completions for the stored prompt set for one model:

```bash
python -m src.dpnlp_pipeline generate-for-prompt-set --model-size small
```

Generate all completions for all configured models and all stored prompts:

```bash
python -m src.dpnlp_pipeline generate-all
```

Create a training loss plot for a trained model:

```bash
python -m src.dpnlp_pipeline plot-loss --model-size xs
```

For a log-scale y-axis, use:

```bash
python -m src.dpnlp_pipeline plot-loss --model-size xs --log-scale
```

Run overlap analysis on generated outputs stored in a JSON file:

```bash
python -m src.dpnlp_pipeline evaluate-overlap --evaluation-json outputs.json
```

Run GPT-based grading on generated outputs stored in a JSON file:

```bash
python -m src.dpnlp_pipeline evaluate-gpt --evaluation-json outputs.json
```

This command requires `OPENAI_API_KEY` to be set and uses the Appendix B GPT-Eval prompt format from the TinyStories paper, including:
- the `***` separator between prompt and completion
- a free-form general assessment
- scores for grammar, creativity, consistency, and plot sense
- an age guess for the student

Build a claim summary report from scored outputs:

```bash
python -m src.dpnlp_pipeline report --evaluation-json scored_outputs.json
```

# What we reproduce
The researchers did many experiments in this paper. For the project we should only reproduce these two:

## Change in Model Capabilities for Different Sizes and Depths
They trained GPT-style models on the dataset with different parameter numbers and different transformer layers. They came to the groundbreaking conclusion that larger models perform better in coherency than smaller models. Shallow models (less transformer layers) often keep grammar but lose story consistency. 

## GPT-eval
Their evaluation pipeline consisted of producing hand-written prompt beginnings and then using their trained models to generate 10 completions per prompt at temperature 1.
GPT-4 was then used to evaluate the completion for grammar, creativity, and consistency with the prompt. 

# Plan
We train 5 models on the TinyStories:
* XS: 1M parameters
* Small: 3M parameters
* Medium: 10M parameters
* Large: 29M parameters
* XL: 56M parameters

Each model is trained for one epoch on the TinyStories Train dataset.

Then we reproduce these claims from the paper:
* Larger TinyStories models produce more coherent story continuations than smaller ones
* Very small models can generate grammatical text before they become good at consistency or creativity
* Generated continuations are not simply memorized from the dataset

## Step-by-step reproduction
1. Create and activate the environment.
   ```bash
   conda env create -f environment.yml
   conda activate chatbot
   ```

2. Confirm that the pipeline is configured correctly and write the default GPT-Eval prompt set.
   ```bash
   python -m src.dpnlp_pipeline describe
   python -m src.dpnlp_pipeline write-prompts
   ```

3. Train all five models for one epoch on `datasets/tinystories/TinyStories-train.txt`.
   ```bash
   python -m src.dpnlp_pipeline train --model-size xs
   python -m src.dpnlp_pipeline train --model-size small
   python -m src.dpnlp_pipeline train --model-size medium
   python -m src.dpnlp_pipeline train --model-size large
   python -m src.dpnlp_pipeline train --model-size xl
   ```

4. Generate story completions from every trained model for the fixed GPT-Eval prompt set.
   The prompts are read from `artifacts/dpnlp_pipeline/gpt_eval_prompts.json`.
   To generate outputs for all models at once, run:
   ```bash
   python -m src.dpnlp_pipeline generate-all
   ```
   This writes one JSON file at:
   `artifacts/dpnlp_pipeline/outputs/all_model_outputs.json`

   If you only want one model, run:
   ```bash
   python -m src.dpnlp_pipeline generate-for-prompt-set --model-size small
   ```
   This writes:
   `artifacts/dpnlp_pipeline/outputs/small_outputs.json`

5. Measure memorization-style overlap against the TinyStories train and validation sets.
   ```bash
   python -m src.dpnlp_pipeline evaluate-overlap --evaluation-json artifacts/dpnlp_pipeline/outputs/all_model_outputs.json
   ```
   Save the result as `overlap_outputs.json`.

6. Run GPT-based grading on the generated outputs.
   Set your OpenAI key first:
   ```bash
   export OPENAI_API_KEY=YOUR_KEY_HERE
   ```
   Then run:
   ```bash
   python -m src.dpnlp_pipeline evaluate-gpt --evaluation-json artifacts/dpnlp_pipeline/outputs/all_model_outputs.json
   ```
   Save the result as `scored_outputs.json`.

7. Merge the GPT scores and overlap scores into one JSON file.
   Each row should contain:
   * `model_size`
   * `prompt`
   * `completion`
   * `general_assessment`
   * `grammar`
   * `creativity`
   * `consistency`
   * `plot_sense`
   * `age_guess`
   * `train_overlap`
   * `valid_overlap`
   * `max_overlap`

8. Build the final claim summary report.
   ```bash
   python -m src.dpnlp_pipeline report --evaluation-json final_scored_outputs.json
   ```

9. Interpret the three claims from the report and the scored outputs.
   * Claim 1 is supported if consistency scores improve from XS through XL.
   * Claim 2 is supported if the smallest models, especially XS and small, have noticeably stronger grammar than creativity and consistency.
   * Claim 3 is supported if generated outputs have low overlap with the training stories while still scoring coherently under GPT-Eval. We also compute validation overlap to be able to do further analytics.

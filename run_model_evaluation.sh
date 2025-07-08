#!/bin/bash

# List of model IDs
model_ids=(
  "openai/whisper-large-v3"
  "syvai/hviske-v2"
  "CoRal-project/roest-whisper-large-v1"
  "CoRal-project/roest-whisper-large-v2c"
  "CoRal-project/roest-wav2vec2-315m-v2"
)

# Loop over each model ID and run the evaluation script
for model_id in "${model_ids[@]}"; do
  echo "Evaluating model: $model_id"
  python src/scripts/evaluate_model.py model_id="$model_id"
done

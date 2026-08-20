#!/bin/bash
source ~/.wandb
echo $WANDB_API_KEY
wandb login $WANDB_API_KEY

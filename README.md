# MyRLJourney
This repository contains many results for sample-efficient Reinforcement Learning on the Atari 100k benchmark.

In addition to studying performance, this repository also contains code and results for policy churn rates, action gaps and generalisation.

To run this code, simply use the requirements.yml files and run "main.py 0 0"
The parameters for main can be used to split runs into multiple jobs, and use different GPUs respectively.

Algorithms and Components Implemented and Respective Median Human-Normalised Performance:

DDQN :white_check_mark: 0.082

DDQN + Image Augmentations :white_check_mark: 0.160

DDQN + Duelling :white_check_mark: 0.075

Data Efficient Rainbow :white_check_mark: 0.155

Self-Predictive Representations :clock3:

SR:SPR :x:

Bigger, Better, Faster :x:

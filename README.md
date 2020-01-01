# StockLearner

## Purpose

1) Use machine learning to predict stock price for China stocks
2) Use reinforcement learning to see how program can play for China stocks

## Requirements

Refer to requirement.txt

### Install requirements

```shell
pip install -r requirements.txt
```

## Configuration file path

1. ./config_file/yaml_config contains model configs and data schema configs
2. tf_keras_sl_ops_***.yaml in root folder contains dataset path, configuration file path
3. specify the yaml in tf_keras_sl_ops.py 

## Run

1. run tf_keras_sl_ops.py to run supervised learning
2. run tf_keras_rl_ops.py to run reinforcement learning 

## Models

Follow keras yaml structure under ./config_file/yaml_config to build model

### Data

- basic_data_schema.yaml and tech_data_schema.yaml contain the structure for datasets
- ./test_data contains some sample dataset 
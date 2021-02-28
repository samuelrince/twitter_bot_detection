# Twitter bot detection from a single tweet

Student project made during the deep learning class at CentraleSupélec. The project is about developing a Twitter bot 
detection model compliant with real time analysis and single tweet data. 

## Intallation

Please first install [poetry](https://python-poetry.org/docs/#installation) package manager in order to install 
dependencies.

Then run the following command to create your python environments and install all the dependencies (python > 3.8.x 
required):

```bash
poetry install
```

## Usage

This package can train 2 types of models, a LSTM-based and a Bert-based. To do training it is required to first 
download the data from [Google Drive](https://drive.google.com/drive/folders/1GvIlQcaZ5IjPzF1ReDq8JrZlFGlT40lb?usp=sharing) 
and put it in the `data` directory. The models are also available in the Google Drive directory.

To train a LSTM please consider the following example commands: 

```bash
python lstm.py \
  --cuda \
  --data ./data \
  --save-dir ./models/lstm \
  --log-interval 500 \
  --epochs 15 \
  --batch-size 32 \
  --seq-len 64 \
  --emsize 200 \
  --nhid 200 \
  --nlayers 2 \
  --dropout 0.5 \
  --lr 0.0001 \
  --clip 5.
```

To train a Bert model:

```bash
python bert.py \
  --cuda \
  --data ./data \
  --save-dir ./models/bert \
  --log-interval 200 \
  --epochs 2 \
  --batch-size 32 \
  --seq-len 64 \
  --lr 0.00001 \
  --eps 0.00000001
```

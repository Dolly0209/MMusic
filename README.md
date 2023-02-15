# MMusic
This repository implements **MMusic** proposed in "MMusic: A hierarchical multi-information fusion method for deep music recommendation" using PyTorch. 
We refer to the model DGRec code [(link)](https://github.com/jbnu-dslab/DGRec-pytorch).


### Installation

This project requires Python 3.8 and the following Python libraries:
- numpy == 1.22.3
- pandas == 1.4.0
- torch == 1.12.0
- fire == 0.4.0
- tqdm == 4.62.3
- loguru == 0.6.0
- dotmap == 1.3.26
  
### Arguments

|Arguments|Explanation|Default|
|------|---|---|
|model|Name of model|'MMusic'|
|data_name|Name of data|'play'|
|seed|Random seed|0|
|epochs|Number of traning epochs |20|
|act|Type of activation function|'relu'|
|batch_size|Size of batch|100|
|learning_rate|Learning rate of model|0.01|
|embedding_size|Size of item and user embedding|50|
|dropout|Dropout rate|0.3|
|decay_rate|weight decay rate|0.98|
|gpu_id|Id of gpu you use|0|


## Data

You can download the raw NOWPLAYINGRS dataset via the [(link)](https://zenodo.org/record/3247476#.Yhnb7ehBybh).

##  Detailed project description
```
main.py: starts training the model
utils.py: contains utility functions
data.py: loads our dataset
MMusic_log.txt: a log file of the results of one of the experiments

src/models/MMusic/eval.py: evaluates the model with a validation dataset
src/models/MMusic/model.py: implements the forward function of our model
src/models/MMusic/train.py: implements a function for training the model

src/models/MMusic/batch/minibatch.py: splits a dataset to mini-batches
```


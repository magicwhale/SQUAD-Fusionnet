# SQUAD

This model is an implementation of the fusionnet model that is described in 
the paper at https://arxiv.org/abs/1711.07341

Requirements before running the model:

Dependencies:

python 3.5 - Python 3.5 must be used for CoVe to work

anaconda   - Used for setting up a python environment

tensorflow - (this model was trained with 1.7.0)

nltk 	   - can be installed with 'sudo pip install -U nltk'

glove 	   - the glove.6B unzipped directory from https://nlp.stanford.edu/projects/glove/

CoVe       - call git clone https://github.com/rgsachin/CoVe.git in the home directory

spacy      - install with pip install -U spacy

spacy english corpus - install with python -m spacy download en

train data - train-v1.1 from the SQUAD website 

dev data   - dev-v1.1 from the SQUAD website

The 'glove.6B/' directory should be placed in the 'data/' directory
'train-v1.1.json' and 'dev-v1.1.json' should be placed in the home directory.
The 'CoVe' folder from github should be placed in the home directory.


Directory Structure:

|-- SQUAD/
|   |-- data/
|   |   |-- glove.6B/
|   |   |   |-- glove.6B.300d.txt
|   |-- CoVe/
|   |-- dev-v1.1.json
|   |-- train-v1.1.json
|   |-- setup.sh
|   |-- main.py
           .
           .
           .


Setup:

Before training, first download and extract glove.6B/ into the data directory, and place train-v1.1 in the home directory.

If all the dependencies are set up already, the rest of the setup section is not needed. All that is needed is to call 

        python preprocess.py

In order to preprocess the data. 

If the dependencies are not set up, run 

        ./setup.sh

This will set up a new Anaconda envirnoment called fusionnet, and start the environment. Before running any code, the environment must first be started by running

        source activate fusionnet

Additionally, setup.sh should download the necessary dependencies and preprocess and load the SQUAD data into the 'data/' directory.

NOTE: By default, ./setup.sh uses Tensorflow without gpu support. If you are running tensorflow with gpu support, change 'pip install tensorflow' to 'pip install tensorflow-gpu' in 'setup.sh'


Training:

To train the model, run:

		python main.py --experiment_name=[EXPERIMENT_NAME] --mode=train

where [EXPERIMENT_NAME] is the name chosen for the training session.
Checkpoints and tensorboard data will then be written to experiments/[EXPERIMENT_NAME]
To view the data, open experiments/[EXPERIMENT_NAME] using tensorboard.


Getting SQUAD score:
To create the file for official f1 evaluation, run:

        python main.py --experiment_name=[EXPERIMENT_NAME] --mode=official_eval --json_in_path=dev-v1.1.json --ckpt_load_dir=experiments/[EXPERIMENT_NAME]

This will create a file named predictions.json

To evaluate the accuracy, run

        python evaluate.py <path_to_dev-v1.1> <path_to_predictions>


Results:

The predicted results for our trained model can be found in the predictions.json file.

SQUAD F1 score (after 25000 training steps): 63.42584912806998
EM score: 52.82876064333018

Credits:
The code structure for this project is loosely based on the code found on the Stanford CS224N website, at http://web.stanford.edu/class/cs224n/
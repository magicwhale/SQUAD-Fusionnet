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
spacy      - install with pip install -U spacy
train data - train-v1.1 from the SQUAD website 
dev data   - dev-v1.1 from the SQUAD website

The 'glove.6B/' directory should be placed in the 'data/' directory
'train-v1.1.json' and 'dev-v1.1.json' should be placed in the home directory

Setup:

Before training, first download and extract glove.6B/ into the data directory, and place train-v1.1 in the home directory.
Then, run 

        ./setup.sh

This will set up a new Anaconda envirnoment called fusionnet, and start the environment. Before running any code, the environment must first be started by running

        source activate fusionnet

Additionally, setup.sh should download the necessary dependencies and preprocess and load the SQUAD data into the 'data/' directory.

NOTE: By default, ./setup.sh uses Tensorflow without the gpu. If you are running tensorflow with the gpu, chang 'pip install tensorflow' to 'pip install tensorflow-gpu'


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

SQUAD F1 score (after 5500 training steps): 28.128770745337675

Credits:
The code structure for this project is loosely based on the code found on the Stanford CS224N website, at http://web.stanford.edu/class/cs224n/
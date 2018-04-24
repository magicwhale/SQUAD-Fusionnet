# SQUAD

This model is an implementation of the fusionnet model that is described in 
the paper at https://arxiv.org/abs/1711.07341

Results:

SQUAD F1 score (after 10):
The model data and checkpoints are found in experiments/firstRun
To get the answer file for the SQUAD dev data using this model, run:

python3 main.py --experiment_name=firstrun --mode=official_eval --json_in_path=dev-v1.1.json --ckpt_load_dir=experiments/firstrun

Requirements before running the model:

Dependencies:
python3	   
tensorflow - (this model was trained with 1.7.0)
nltk 	   - can be installed with 'sudo pip install -U nltk'
glove 	   - the glove.6B unzipped directory from https://nlp.stanford.edu/projects/glove/
train data - train-v1.1 from the SQUAD website 
dev data   - dev-v1.1 from the SQUAD website

The 'glove.6B/' directory should be placed in the 'data/' directory
'train-v1.1.json' and 'dev-v1.1.json' should be placed in the home directory

Setup:
Before running the main file, run 'python preprocess.py' to preprocess and load the SQUAD data.
The data should appear in the 'data/' directory.


Training:
To train the model, run:

		python3 main.py --experiment_name=[EXPERIMENT_NAME] --mode=train

where [EXPERIMENT_NAME] is the name chosen for the training session.
Checkpoints and tensorboard data will then be written to experiments/[EXPERIMENT_NAME]
To view the data, open experiments/[EXPERIMENT_NAME] using tensorboard.


Getting SQUAD score:
To create the file for official f1 evaluation, run:

python main.py --experiment_name=[EXPERIMENT_NAME] --mode=official_eval --json_in_path=dev-v1.1.json --ckpt_load_dir=experiments/[EXPERIMENT_NAME]

To evaluate the accuracy, run
python evaluate.py <path_to_dev-v1.1> <path_to_predictions>

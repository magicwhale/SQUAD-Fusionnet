import tensorflow as tf
import pickle
import os
from load import loadGlove
from batch import batch, generateBatches
from model import Model

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir

# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size", 200, "Size of the hidden states")
tf.app.flags.DEFINE_integer("context_len", 600, "The maximum context length of your model")
tf.app.flags.DEFINE_integer("question_len", 30, "The maximum question length of your model")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")
tf.app.flags.DEFINE_integer("fusion_size", 250, "Size of hidden fusion layer")

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")


FLAGS = tf.app.flags.FLAGS

def loadDataFiles(dataDir, dataName):
    with open(os.path.join(dataDir, dataName +'.context'), 'rb') as contextFile,  \
         open(os.path.join(dataDir, dataName +'.question'), 'rb') as questionFile,  \
         open(os.path.join(dataDir, dataName +'.answer'), 'rb') as answerFile,  \
         open(os.path.join(dataDir, dataName +'.span'), 'rb') as spanFile:

        contexts = pickle.load(contextFile)
        questions = pickle.load(questionFile)
        answers = pickle.load(answerFile)
        spans = pickle.load(spanFile)
        return contexts, questions, answers, spans

def main(argv):
    contexts, questions, answers, spans = loadDataFiles('data', 'training')
    wordToId, idToWord, embMat = loadGlove("data/glove.6b", 300)
    # model = Model(FLAGS, wordToId, idToWord, embMat)
    # print(wordToId)
    # print(answers)
    # if FLAGS.mode == "train":
    with tf.Session() as sess:
        
    # generateBatches(wordToId, contexts, questions, spans, 200)

if __name__ == "__main__":
    tf.app.run()
import tensorflow as tf
import pickle
import os
import json
import logging
from load import loadGlove
from load import extractCtxtQn
from load import loadJsonData
from load import findAnswers
from batch import batch, generateBatches
from model import Model

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))# relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir

# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.3, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use")
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

def loadGloveFiles(dataDir):
    with open(os.path.join(dataDir, 'glove.w2i'), 'rb') as w2i_file,  \
         open(os.path.join(dataDir, 'glove.i2w'), 'rb') as i2w_file,  \
         open(os.path.join(dataDir, 'glove.embMat'), 'rb') as mat_file:

        wordToId = pickle.load(w2i_file)
        idToWord = pickle.load(i2w_file)
        gloveMat = pickle.load(mat_file)
        return wordToId, idToWord, gloveMat

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
      
def loadModel(session, model, checkpointDir, modelExpected):
    checkpoint = tf.train.get_checkpoint_state(checkpointDir)

    # load checkpoint if it exists
    if checkpoint:
        model.saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        if modelExpected:
            raise Exception("No checkpoints in %s" % checkpointDir)
        else:
            session.run(tf.global_variables_initializer())
      
      
def main(argv):

    if not FLAGS.experiment_name and not FLAGS.train_dir and FLAGS.mode != "official_eval":
        raise Exception("You need to specify either --experiment_name or --train_dir")
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)

    print("loading train data")
    trainContexts, trainQuestions, trainAnswers, trainSpans = loadDataFiles('data', 'training')
    print("loading dev data")
    devContexts, devQuestions, devAnswers, devSpans = loadDataFiles('data', 'dev')
    print("loading glove data")
    wordToId, idToWord, gloveMat = loadGloveFiles('data')
    # model = Model(FLAGS, wordToId, idToWord, embMat)
    # print(wordToId)
    # print(answers)
    # if FLAGS.mode == "train":
    #     with tf.Session() as sess:
    # generateBatches(wordToId, contexts, questions, spans, 200)

    # prepare directory for best model

    bestDir = os.path.join(FLAGS.train_dir, "best_checkpoint")
    model = Model(FLAGS, wordToId, idToWord, gloveMat)

    # GPU settings
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # modes
    if FLAGS.mode == "train":
        # Setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # save flags
        with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as flag_rec:
            # json.dump(FLAGS.__flags, flag_rec)
            flag_rec.write(str(FLAGS.__dict__['__wrapped']))

        # Make bestmodel dir if necessary
        if not os.path.exists(bestDir):
            os.makedirs(bestDir)

        with tf.Session() as sess:
            loadModel(sess, model, FLAGS.train_dir, modelExpected=False)
            model.train(sess, trainContexts, trainQuestions, trainSpans, devContexts, devQuestions, devSpans) 


    elif FLAGS.mode == "show_examples":
        with tf.Session() as sess:
            loadModel(sess, model, FLAGS.train_dir, modelExpected=True)
            # TODO: F1 scores


    elif FLAGS.mode == "official_eval":
        if FLAGS.json_in_path == "":
            raise Exception("Need to specify json data path")
        if FLAGS.ckpt_load_dir == "":
            raise Exception("Need to specify checkpoint directory")

        data = loadJsonData(FLAGS.json_in_path)
        quesIdSet, contexts, questions = extractCtxtQn(data)

        with tf.Session() as sess:
            loadModel(sess, model, FLAGS.ckpt_load_dir, modelExpected=True)
            answers = findAnswers(sess, model, wordToId, quesIdSet, contexts, questions)
            with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as out:
                out.write(str(json.dumps(answers, ensure_ascii=False)))  

    else:
        raise Exception("Given mode does not exist")

            

if __name__ == "__main__":
    tf.app.run()

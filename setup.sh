conda create -n fusionnet python=3.5

source activate fusionnet

pip install keras
pip install tensorflow #use tensorflow-gpu if using gpu
pip install numpy
pip install nltk

python -m nltk.downloader punkt
python -m nltk.downloader perluniprops

pip install -U spacy

git clone https://github.com/rgsachin/CoVe.git

python3 preprocess.py

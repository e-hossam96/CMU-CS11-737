wget https://raw.githubusercontent.com/neulab/word-embeddings-for-nmt/master/ted_reader.py

mkdir -p data
cd data

wget http://phontron.com/data/ted_talks.tar.gz
tar -xvf ted_talks.tar.gz

cd ..

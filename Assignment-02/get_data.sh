mkdir -p data
cd data

wget http://phontron.com/data/ted_talks.tar.gz
tar -xvf ted_talks.tar.gz

cd ..

python ted_reader.py
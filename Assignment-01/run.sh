LANG_CODES="en es cs ar af lt hy ta"

for lang in $LANG_CODES
do
    echo "Training and Testing on the $lang language .."
    python main.py --mode train --lang $lang
    python main.py --mode eval --lang $lang
    echo "---------------------------------"
    echo "---------------------------------"
done

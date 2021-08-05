
echo '********HyperParams*******'

# py classifier.py --gpu 1 --max_epochs 5 --BATCH_SIZE 16 --cls 'ca'
LR=1e-4
EPOCHS=12
BATCH_SIZE=16
seed=-1
fold=-1


echo "LR : $LR"
echo "EPOCHS : $EPOCHS"
echo "BATCH_SIZE : $BATCH_SIZE"

echo '**************************'

for i in $(seq 0 200);
do

    python3 classifier.py  \
        --gpu 1 \
        --max_epochs $EPOCHS\
        --BATCH_SIZE $BATCH_SIZE\
        --cls 'c' \
        --i $i \
        --seed $seed \
        --fold $fold
done
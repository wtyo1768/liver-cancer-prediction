
echo '********HyperParams*******'

LR=1e-4
EPOCHS=10
BATCH_SIZE=16
seed=46
fold=-1

echo "LR : $LR"
echo "EPOCHS : $EPOCHS"
echo "BATCH_SIZE : $BATCH_SIZE"

echo '**************************'

for i in $(seq 0 10);
do
    python3 CGC.py  \
        --gpu 1 \
        --max_epochs $EPOCHS\
        --BATCH_SIZE $BATCH_SIZE\
        --cls 'single' \
        --i $i \
        --seed $seed \
        --fold $fold
done
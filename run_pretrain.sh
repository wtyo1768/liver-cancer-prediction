

echo '********HyperParams*******'

#23 46
LR=3e-4
EPOCHS=12
BATCH_SIZE=32

echo "LR : $LR"
echo "EPOCHS : $EPOCHS"
echo "BATCH_SIZE : $BATCH_SIZE"

echo '**************************'

for i in $(seq 0 10);
do

    python3 ssl_byol.py \
        --LR $LR \
        --EPOCHS $EPOCHS \
        --BATCH_SIZE $BATCH_SIZE\
        --i $i 
done
echo "----------- Batch ---------------"
python batch_mnistNN.py --error-file 'batch_error_alpha03_batch2_epochs3.pickle' --alpha .03 --batch-size 2 --num-epochs 3

python mnistNN.py --error-file 'online_error_alpha03_batch2_epochs3.pickle' --alpha .03 --batch-size 2 --num-epochs 3

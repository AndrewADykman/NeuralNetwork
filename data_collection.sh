echo "----------- Batch ---------------"
python batch_mnistNN.py --error-file 'batch_error_alpha06_batch200_epochs20.pickle' --alpha .06 --batch-size 200 --num-epochs 20

python batch_mnistnn.pt --error-file 'batch_error_alpha06_batch100_epochs20.pickle '--alpha .06 --batch-size 100 --num-epochs 20

python batch_mnistnn.pt --error-file 'batch_error_alpha06_batch1000_epochs20.pickle '--alpha .06 --batch-size 1000 --num-epochs 20

python batch_mnistnn.pt --error-file 'batch_error_alpha06_batch500_epochs20.pickle '--alpha .06 --batch-size 500 --num-epochs 20

python batch_mnistnn.pt --error-file 'batch_error_alpha06_batch50_epochs20.pickle '--alpha .06 --batch-size 50 --num-epochs 20

python batch_mnistnn.pt --error-file 'batch_error_alpha06_batch5_epochs20.pickle '--alpha .06 --batch-size 5 --num-epochs 20

echo "---------- Online ---------------"
python mnistNN.py --error-file 'online_error_alpha01_epochs20.pickle' --alpha .01 --num-epochs 20

python mnistNN.py --error-file 'online_error_alpha03_epochs20.pickle' --alpha .03 --num-epochs 20

python mnistNN.py --error-file 'online_error_alpha06_epochs20.pickle' --alpha .06 --num-epochs 20

python mnistNN.py --error-file 'online_error_alpha10_epochs20.pickle' --alpha .10 --num-epochs 20

python mnistNN.py --error-file 'online_error_alpha20_epochs20.pickle' --alpha .20 --num-epochs 20

python mnistNN.py --error-file 'online_error_alpha001_epochs20.pickle' --alpha .001 --num-epochs 20


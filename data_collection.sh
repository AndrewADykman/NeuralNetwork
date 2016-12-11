echo "----------- Mini-Batch Size Experiment ---------------"

echo 'zero. Zero indexed by mistake, but its staying this way'

python batch_mnistNN.py --error-file 'batch_error_alpha06_batch200_epochs20.pickle' --alpha .06 --batch-size 200 --num-epochs 20

echo 'one'

python batch_mnistnn.pt --error-file 'batch_error_alpha06_batch100_epochs20.pickle '--alpha .06 --batch-size 100 --num-epochs 20

echo 'two'

python batch_mnistnn.pt --error-file 'batch_error_alpha06_batch1000_epochs20.pickle '--alpha .06 --batch-size 1000 --num-epochs 20

echo 'three'

python batch_mnistnn.pt --error-file 'batch_error_alpha06_batch500_epochs20.pickle '--alpha .06 --batch-size 500 --num-epochs 20

echo 'four'

python batch_mnistnn.pt --error-file 'batch_error_alpha06_batch50_epochs20.pickle '--alpha .06 --batch-size 50 --num-epochs 20

echo 'five'

python batch_mnistnn.pt --error-file 'batch_error_alpha06_batch5_epochs20.pickle '--alpha .06 --batch-size 5 --num-epochs 20

echo "---------- Online Alpha Experimentation---------------"

echo 'zero'

python mnistNN.py --error-file 'online_error_alpha01_epochs20.pickle' --alpha .01 --num-epochs 20

echo 'one'

python mnistNN.py --error-file 'online_error_alpha03_epochs20.pickle' --alpha .03 --num-epochs 20

echo 'two'

python mnistNN.py --error-file 'online_error_alpha06_epochs20.pickle' --alpha .06 --num-epochs 20

echo 'three'

python mnistNN.py --error-file 'online_error_alpha10_epochs20.pickle' --alpha .10 --num-epochs 20

echo 'four'

python mnistNN.py --error-file 'online_error_alpha20_epochs20.pickle' --alpha .20 --num-epochs 20

echo 'five'

python mnistNN.py --error-file 'online_error_alpha001_epochs20.pickle' --alpha .001 --num-epochs 20


echo "----------- Full-Batch Size Experiment ---------------"

echo 'zero. Zero indexed by mistake, but its staying this way'

python batch_mnistNN.py --error-file 'batch_error_alpha05_batchFULL_epochs20.pickle' --alpha .05 --batch-size -1 --num-epochs 20

echo 'one'

python batch_mnistnn.pt --error-file 'batch_error_alpha01_batchFULL_epochs20.pickle '--alpha .1 --batch-size -1 --num-epochs 20

echo 'two'

python batch_mnistnn.pt --error-file 'batch_error_alpha20_batchFULL_epochs20.pickle '--alpha .2 --batch-size -1 --num-epochs 20

echo 'three'

python batch_mnistnn.pt --error-file 'batch_error_alpha40_batchFULL_epochs20.pickle '--alpha .4 --batch-size -1 --num-epochs 20

echo 'four'

python batch_mnistnn.pt --error-file 'batch_error_alpha01_batchFULL_epochs20.pickle '--alpha .01 --batch-size -1 --num-epochs 20

echo 'five'

python batch_mnistnn.pt --error-file 'batch_error_alpha06_batchFULL_epochs20.pickle '--alpha .6 --batch-size -1 --num-epochs 20

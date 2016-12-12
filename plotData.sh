for D in ./pickles/*.pickle
do
    currfile=$(basename $D)
    pref="./pickle/"
    echo "$D"
    python plotData.py $D
done

DATADIR=/media/memory/noe/databases/EmoV-DB_sorted_trimmed
CODEDIR=/home/noetits/doctorat_code/ophelia
cd $DATADIR
for d in */; do
    echo $d
    cd $d
    for e in */; do
        echo $e
        python $CODEDIR/script/normalise_level.py -i $DATADIR/$d/$e -o $DATADIR/wav_norm/$d -ncores 25
    done
cd ..
done

cd $DATADIR/wav_norm
WAVDIR=$DATADIR/wav_final
mkdir $WAVDIR

for d in *; do
    echo $d
    cd $d
    for f in *.wav; do
    echo $d-$f
    cp $DATADIR/wav_norm/$d/$f $WAVDIR/$d-$f
    done
    cd ..
done
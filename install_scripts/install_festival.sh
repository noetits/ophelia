INSTALL_DIR=$1

mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/festival-2.4-release.tar.gz
wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/speech_tools-2.4-release.tar.gz

tar xvf festival-2.4-release.tar.gz
tar xvf speech_tools-2.4-release.tar.gz

## Install Speech tools first
cd speech_tools
./configure  --prefix=$INSTALL_DIR
gmake

## Then compile Festival
cd ../festival
./configure  --prefix=$INSTALL_DIR
gmake

# Finally, get a Festival voice with the CMU lexicon
cd ..
wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/voices/festvox_cmu_us_awb_cg.tar.gz
tar xvf festvox_cmu_us_awb_cg.tar.gz
wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_CMU.tar.gz
tar xvf festlex_CMU.tar.gz
wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_POSLEX.tar.gz
tar xvf festlex_POSLEX.tar.gz

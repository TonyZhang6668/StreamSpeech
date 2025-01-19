lang=es # Language code (eg. Spanish), which will be passed to all scripts as an argument. Set lang variable to Spanish.

ROOT=/data/zhangshaolei/StreamSpeech
PREPROCESS_ROOT=$ROOT/preprocess_scripts # Specify the path.

# 1. Extracts features from the 11th layer of a HuBERT-based model. 
# 2. Applies K-means clustering to those features. 
# 3. Converts feature vectors into discrete K-means cluster indices. 
# 4. Produces quantized features for train, dev, and test sets for the Spanish-English CVSS-C dataset. --save in dictionary('audio ID'...)
bash $PREPROCESS_ROOT/1.learn_KM_clustering_model.sh $lang
echo 'finish 1.learn_KM_clustering_model.sh'

# 1. Combines CVSS-C and CoVoST2 data, contains cross-lingual paired speech data (CoVoST2 propvides more variety). ---take input from human speech to text, mapping to CVSS(robot speech)
# 2. Converts speech from source wave (Spanish raw audio) to unit-based representations classified in K-means 1000-unit system.
# 3. Uses a HiFi-GAN vocoder to reconstruct waveforms from the unit sequences (eg. Model can take unit sequences (like 101, 22, 56, 345) and generate raw waveform audio).
bash $PREPROCESS_ROOT/2.prep_cvss_c_multilingual_data.sh $lang
echo 'finish 2.prep_cvss_c_multilingual_data.sh'

# 1. Source language data in fbank2unit format, which includes filterbank features and discrete units.
# 2. Converts the unit files into tokenized subword units using a unigram vocabulary with 6000 tokens. --> for multitask learning
# 3. Outputs tokenized train, dev, and test datasets in subword format and a vocabulary file for the tokenized subword units.
bash $PREPROCESS_ROOT/3.prep_cvss_c_multitask_data.sh $lang
echo 'finish 3.prep_cvss_c_multitask_data.sh'

# Prepares the reference text from source text for the CVSS.
bash $PREPROCESS_ROOT/5.prep_cvss_c_ref_txt.sh $lang
echo 'finish 5.prep_cvss_c_ref_txt.sh'

# Extracts data into a form compatible with SimulEval for later evaluating.
bash $PREPROCESS_ROOT/6.extract_simuleval_data.sh $lang
echo 'finish 6.extract_simuleval_data.sh'

# Prepares data for ASR multitask.
bash $PREPROCESS_ROOT/7.prep_cvss_c_multitask_asr_data.sh $lang
echo 'finish 7.prep_cvss_c_multitask_asr_data.sh'

# Outputs unit (speech) and source (text) data files for training.
bash $PREPROCESS_ROOT/8.prep_cvss_c_simuleval_unit.sh $lang
bash $PREPROCESS_ROOT/8.prep_cvss_c_simuleval_src.sh $lang
echo 'finish 8.prep_cvss_c_simuleval_unit.sh, 8.prep_cvss_c_simuleval_src.sh '

# # only for s2tt training on CVSS-C
# bash $PREPROCESS_ROOT/9.prep_cvss_c_s2st_mtl_data.sh  $lang
# echo 'finish 9.prep_cvss_c_s2st_mtl_data.sh'


# input, output paths --> step to step
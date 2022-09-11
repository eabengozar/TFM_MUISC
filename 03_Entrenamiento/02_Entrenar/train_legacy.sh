# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH="/home/eugenio/TFM_MUISC_640x480_02/Scripts/02_Entrenar/legacy_01/model_tfm_resnet.config"
TRAIN_DIR="/home/eugenio/TFM_MUISC_640x480_02/models/model_legacy_01/train/"
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 /home/eugenio/models/research/object_detection/legacy/train.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --train_dir=${TRAIN_DIR} 


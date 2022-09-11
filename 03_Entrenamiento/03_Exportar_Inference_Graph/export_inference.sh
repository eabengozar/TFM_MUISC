# From tensorflow/models/research/
#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md
# para pasar del ckpt al .pb
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH="/home/eugenio/TFM_MUISC_640x480_02/Scripts/02_Entrenar/model_tfm_resnet.config"
TRAINED_CKPT_PREFIX="/home/eugenio/TFM_MUISC_640x480_02/models/model_legacy_01/train/model.ckpt-75490"
EXPORT_DIR="/home/eugenio/TFM_MUISC_640x480_02/models/model_legacy_01/export_inference"
python3 /home/eugenio/models/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}


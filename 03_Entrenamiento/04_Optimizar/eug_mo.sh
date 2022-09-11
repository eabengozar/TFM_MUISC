SHAPE=[1,224,224,3]
INPUT_MODEL="/home/eugenio/TFM_MUISC_640x480_02/models/model_legacy_01/export_inference/frozen_inference_graph.pb"
CAPA_ENTRADA=image_tensor
JSON_MODELO="/opt/intel/openvino_2021/deployment_tools/model_optimizer/extensions/front/tf/mask_rcnn_support_api_v1.15.json "
PIPELINE_CONFIG="/home/eugenio/TFM_MUISC_640x480_02/models/model_legacy_01/export_inference/pipeline.config"
OUTPUT_DIR="/home/eugenio/TFM_MUISC_640x480_02/models/model_legacy_01/modelo_Convertido/"
#	--keep_shape_ops \

python3 /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py \
	--input_shape ${SHAPE} \
	--input=image_tensor \
	--input_model ${INPUT_MODEL}  \
	--reverse_input_channels \
	--tensorflow_use_custom_operations_config=${JSON_MODELO} \
	--tensorflow_object_detection_api_pipeline_config ${PIPELINE_CONFIG} \
	--data_type FP16 \
	--output_dir ${OUTPUT_DIR}

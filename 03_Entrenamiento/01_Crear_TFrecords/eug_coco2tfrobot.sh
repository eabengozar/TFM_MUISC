#!/bin/bash
#    python create_coco_tf_record.py --logtostderr \
#      --train_image_dir="${TRAIN_IMAGE_DIR}" \
#      --val_image_dir="${VAL_IMAGE_DIR}" \
#      --test_image_dir="${TEST_IMAGE_DIR}" \
#      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
#      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
#      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
#      --output_dir="${OUTPUT_DIR}"

TRAIN_IMAGE_DIR="/home/eugenio/TFM_MUISC_640x480_02/dataset/train"
VAL_IMAGE_DIR="/home/eugenio/TFM_MUISC_640x480_02/dataset/val"
TEST_IMAGE_DIR="/home/eugenio/TFM_MUISC_640x480_02/dataset/test"
TRAIN_ANNOTATIONS_FILE="/home/eugenio/TFM_MUISC_640x480_02/dataset_json/train.json"
VAL_ANNOTATIONS_FILE="/home/eugenio/TFM_MUISC_640x480_02/dataset_json/val.json"
TESTDEV_ANNOTATIONS_FILE="/home/eugenio/TFM_MUISC_640x480_02/dataset_json/test.json"
OUTPUT_DIR="/home/eugenio/TFM_MUISC_640x480_02/tfrecords"

    python3 create_coco_tf_record_eug.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --include_masks=True \
      --output_dir="${OUTPUT_DIR}"

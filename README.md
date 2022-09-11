# TFM_MUISC

Este repositorio ha sido creado como complemento a la memoria del Trabajo de Fin del Máster Universitario en Ingeniería de Sistemas y Control, y recoge todos los programas, imágenes e información adicional generados para su desarrollo.

A continuación se listan los diferentes materiales que se encuentran en cada directorio.

## 01_Calibración_de_la_cámara
- **01_camera_calibration.py:** Este script realiza la calibración de la cámara a partir de un conjunto de imágenes de calibración. Está realizado a partir de la documentación de OpenCV.
- **02_test_camera_calibration.py:** Script para comprobar la calibración realizada sobre otro conjunto de imágenes.

## 02_Generación_del_dataset:
- **01_dataset_generator.py:** Genera los conjuntos de Entrenamiento, Validación y Test creando proyecciones de la imagen base sobre distintas imágenes de fondos. Además de las imágenes genera las máscaras y guarda las poses con las que han sido
creadas.
- **02_create-custom-coco-dataset.py:** Genera las anotaciones COCO a partir de las máscaras del paso anterior.

## 03_Entrenamiento:
- 01_Crear_Tfrecords:
  - **create_coco_tf_record_eug.py:** Script de Object Detection API en el que tan solo se han especificado el número de imágenes de los distintos conjuntos (en las líneas 494, 504 y 514).
  - **eug_coco2tfrobot.sh:** Script de Shell para la ejecución más cómoda del script
anterior con todos los argumentos necesarios.

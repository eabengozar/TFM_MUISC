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

- 02_Entrenar:
  - **model_tfm_resnet.config:** Es el archivo de configuración empleado para el en- trenamiento, el que se detalla en el Anexo A
  - **train_legacy.sh:** Script de Shell que lanza el entrenamiento mediante Object Detection API con toda la parametrización necesaria.
  - **eval_legacy.sh:** Script de Shell para la validación a través de Object Detection API. Es necesario lanzar este Script junto al de entrenamiento, para que la evaluación se haga durante todos los puntos de control del entrenamiento. Para que no existan conflictos con el uso de la GPU se ha modificado la llamada al script de evaluación para que se ejecute en la CPU.
  - **test_legacy.sh:** Se trata de un script de evaluación como el anterior, pero modificado para que cargue un conjunto de Test y evalúe el rendimiento del modelo. También necesita el archivo de configuración modificado a partir del descrito en el Anexo A, model_tfm_resnet_test.config.

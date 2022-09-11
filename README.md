# TFM_MUISC

Este repositorio ha sido creado como complemento a la memoria del Trabajo de Fin del Máster Universitario en Ingeniería de Sistemas y Control, y recoge todos los programas, imágenes e información adicional generados para su desarrollo.

A continuación se listan los diferentes programas que se encuentran en cada directorio.

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

- 03_Exportar_Inferencia:
  - **export_inference.sh:** Este script de shell se encarga de la parametrización necesaria para export_inference_graph.py, herramienta de Object Detection API para preparar el modelo entrenado para poder utilizarlo en inferencias.

- 04_Optimizar:
  - **eug_mo.sh:** Script de shell que parametriza la llamada a mo_tf.py, herramienta para optimizar el modelo de inferencia de TensorFlow para que pueda utilizarse en el entorno Openvino.

## 04_Estimación_de_la_pose:
- **01_Estimacion_poses_test_rgchromaticy.py:** Estima la pose de la imagen base en el conjunto de Test a partir de la imagen y la máscara. Genera un archivo con todas las poses.
- **02_comparativa_poses.py:** Compara las poses generadas con el dataset con las estimadas en el paso anterior, guarda los errores en un archivo y dibuja un gráfico de barras con ellos.

## 05_Integración_del_sistema_de_visión:
- catkin_tfm_MUISC: Directorio catkin de ROS con los paquetes de los dos nodos necesarios:
  - **ros_openvino2021:** Nodo para la inferencia Openvino modificado para que uno de los tópics de salida sea la máscara resultado de la inferencia.
  - **estima_pose:** Nodo que estima la pose a partir de los topics de la imagen original y la máscara.

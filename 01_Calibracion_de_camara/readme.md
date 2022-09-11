
- El directorio **Imgs_Calibracion** contiene las imágenes utilizadas para calibración. Para probar distintas opciones se han preparado imágenes en dos resoluciones de la cámara, 1280x720 y 640x480. 
- El directorio **test** contiene otras imágenes para probar que la calibración ha sido correcta.
- En **01_camera_calibration.py** puede configurarse el directorio del que toma las imágenes para hacer la calibración.
- En **02_test_camera_calibration.py** pueden configurarse tanto el directorio como la resolución (la matriz intrínseca y los coeficientes de distorsión son distintos).
- Desarrollado y probado en **Python 3.8.10**

# A partir del código de la página https://stackoverflow.com/questions/6606891/opencv-virtually-camera-rotating-translating-for-birds-eye-view

import cv2
import numpy as np
import random
import os
import csv


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# número de imágenes a generar de los distintos conjuntos
n_train = 8
n_val = 1
n_test = 1

# imagen base con la que crear el dataset
imagen_base = './imagen_base_s.jpg'

# resolucion de la cámara
resolucion = '640x480' #'640x480' #'1280x720'

# imágenes del fondo
fondo_lab = './Fondo_Lab/Data_{}/'.format(resolucion)

#directorios de entrenamiento, validación y test
dir_train = './dataset/train/'
dir_train_mask = './dataset/train_mask/'
dir_val = './dataset/val/'
dir_val_mask = './dataset/val_mask/'
dir_test = './dataset/test/'
dir_test_mask = './dataset/test_mask/'
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def gen_proyeccion(figura,fondo):

	dst = np.zeros_like(fondo)
	h_f, w_f = figura.shape[:2]
	h_b, w_b = fondo.shape[:2]

	# desplazamiento y orientaciones aleatorias dentro de rangos
	# los rangos se podrían parametrizar
	distZ = random.randrange(2000,5000)
	distX = random.randrange(-1300,1000)
	distY = random.randrange(-500,800)
	rotXval = random.randrange(30,150)
	rotYval = random.randrange(30,150)
	rotZval = random.randrange(30,150)

	rotX = (rotXval - 90)*np.pi/180
	rotY = (rotYval - 90)*np.pi/180
	rotZ = (rotZval - 90)*np.pi/180

	# Matriz intrínseca de la cámara y Matriz de Proyección 2D -> 3D
	if resolucion=='1280x720':
		#1280x720
		K = np.array([[962, 0, 667, 0],
			    [0, 964, 339, 0],
			    [0, 0,   1, 0]])
		A1 = np.array([[1, 0, -667/2+250],
				[0, 1, -339/2+100],
				[0, 0, 0],	
				[0, 0,   1]])
	elif resolucion=='640x480':
		#640x480
		K = np.array([[634, 0, 344, 0],
	 		[  0, 634,  232, 0],
	 		[  0,  0,  1  , 0]])
		A1 = np.array([[1, 0, -344/2+125],
			[0, 1, -232/2+50],
			[0, 0, 0],	
			[0, 0,   1]])	

	# Matrices de rotación sobre los ejes X,Y,Z
	RX = np.array([[1,           0,            0, 0],
		    [0,np.cos(rotX),-np.sin(rotX), 0],
		    [0,np.sin(rotX),np.cos(rotX) , 0],
		    [0,           0,            0, 1]])

	RY = np.array([[ np.cos(rotY), 0, np.sin(rotY), 0],
		    [            0, 1,            0, 0],
		    [ -np.sin(rotY), 0, np.cos(rotY), 0],
		    [            0, 0,            0, 1]])

	RZ = np.array([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
		    [ np.sin(rotZ), np.cos(rotZ), 0, 0],
		    [            0,            0, 1, 0],
		    [            0,            0, 0, 1]])

	# Composición de las rotaciones
	R = np.linalg.multi_dot([ RX , RY , RZ ])

	# Matriz de traslación
	T = np.array([[1,0,0,distX],
		    [0,1,0,distY],
		    [0,0,1,distZ],
		    [0,0,0,1]])

	# Matriz homográfica
	H = np.linalg.multi_dot([K, T, R, A1])

	# Aplicamos la transformación sobre la figura
	cv2.warpPerspective(figura, H, (w_b, h_b), dst, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)

	# aplicamos la transformación a las esquinas para crear la máscara para la fusión
	corners_bef = np.float32([[0, 0], [w_f, 0], [w_f, h_f], [0, h_f]]).reshape(-1, 1, 2)
	corners_aft = cv2.perspectiveTransform(corners_bef, H)

	#creamos la máscara
	mask = np.zeros(fondo.shape[:2], np.uint8)		
	pts = np.around(corners_aft)
	pts = pts.astype(int)		
	cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

	mask2 = cv2.bitwise_and(fondo, fondo, mask=mask)
	mask2 = cv2.bitwise_not(mask2) #invertimos negro y blanco

	#fondo con máscara negra en la figura
	fondo_masked = cv2.bitwise_and(fondo,mask2)

	#fusión de las imágenes de figura y fondo
	final =  cv2.bitwise_or(dst, fondo_masked)

	return final,mask,R,T


if __name__ == '__main__':

	#Leemos la imagen base
	base = cv2.imread(imagen_base)
	
	# usaremos el HSV para adaptar la luminosidad de la imagen base sobre el fondo
	hsv_figura = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
	hue_figura,sat_figura,val_fig = cv2.split(hsv_figura)
	
	#generamos el fichero de anotaciones de poses
	anotation = open(dir_train+'poses.txt', 'w')
	writer = csv.writer(anotation)
	header = ['imagen','rotacion','traslacion']
	writer.writerow(header)

	for i in range(n_train):

		path = fondo_lab+'train/'
		file = random.choice(os.listdir(path))
		fondo = cv2.imread(path+file)
		
		#adaptamos la luminancia de la figura al fondo
		hsv_fondo = cv2.cvtColor(fondo, cv2.COLOR_BGR2HSV)
		hue,sat,val_fondo = cv2.split(hsv_fondo)

		factor = 1.0 #para darle un poquito más de luz
		val_figura = np.trunc(val_fig*np.average(val_fondo)*factor/np.average(val_fig))
		val_figura = val_figura.astype(np.uint8)
		hsv_figura = cv2.merge([hue_figura, sat_figura, val_figura])
		src = cv2.cvtColor(hsv_figura, cv2.COLOR_HSV2BGR)

		final,mask,R,T = gen_proyeccion(src,fondo)
		
		row = [str(i).zfill(4) + '.jpg', R, T]
		writer.writerow(row)
		
		#cv2.imshow(wndname1, final)
		#cv2.imshow(wndname2, mask)
		cv2.imwrite(dir_train + str(i).zfill(4) + '.jpg',final)
		cv2.imwrite(dir_train_mask + str(i).zfill(4) + '.jpg',mask)
		
		#k = cv2.waitKey(0)


	anotation.close()
	#generamos el fichero de anotaciones de poses
	anotation = open(dir_val+'poses.txt', 'w')
	writer = csv.writer(anotation)
	header = ['imagen','rotacion','traslacion']
	writer.writerow(header)
	for i in range(n_val):
	
		path = fondo_lab+'val/'
		file = random.choice(os.listdir(path))
		fondo = cv2.imread(path+file)
		
		#adaptamos la luminancia de la figura al fondo
		hsv_fondo = cv2.cvtColor(fondo, cv2.COLOR_BGR2HSV)
		hue,sat,val_fondo = cv2.split(hsv_fondo)

		factor = 1.0 #para darle un poquito más de luz
		val_figura = np.trunc(val_fig*np.average(val_fondo)*factor/np.average(val_fig))
		val_figura = val_figura.astype(np.uint8)
		hsv_figura = cv2.merge([hue_figura, sat_figura, val_figura])
		src = cv2.cvtColor(hsv_figura, cv2.COLOR_HSV2BGR)

		final,mask,R,T = gen_proyeccion(src,fondo)
		
		row = [str(i).zfill(4) + '.jpg', R, T]
		writer.writerow(row)
		
		cv2.imwrite(dir_val + str(i).zfill(4) + '.jpg',final)
		cv2.imwrite(dir_val_mask + str(i).zfill(4) + '.jpg',mask)
		
		
	anotation.close()
	#generamos el fichero de anotaciones de poses
	anotation = open(dir_test+'poses.txt', 'w')
	writer = csv.writer(anotation)
	header = ['imagen','rotacion','traslacion']
	writer.writerow(header)
	for i in range(n_test):
	
		path = fondo_lab+'test/'
		file = random.choice(os.listdir(path))
		fondo = cv2.imread(path+file)
		
		#adaptamos la luminancia de la figura al fondo
		hsv_fondo = cv2.cvtColor(fondo, cv2.COLOR_BGR2HSV)
		hue,sat,val_fondo = cv2.split(hsv_fondo)

		factor = 1.0 #para darle un poquito más de luz
		val_figura = np.trunc(val_fig*np.average(val_fondo)*factor/np.average(val_fig))
		val_figura = val_figura.astype(np.uint8)
		hsv_figura = cv2.merge([hue_figura, sat_figura, val_figura])
		src = cv2.cvtColor(hsv_figura, cv2.COLOR_HSV2BGR)

		final,mask,R,T = gen_proyeccion(src,fondo)
		
		row = [str(i).zfill(4) + '.jpg', R, T]
		writer.writerow(row)
		
		cv2.imwrite(dir_test + str(i).zfill(4) + '.jpg',final)
		cv2.imwrite(dir_test_mask + str(i).zfill(4) + '.jpg',mask)
		
		

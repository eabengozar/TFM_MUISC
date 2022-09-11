import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import NamedTuple
from statistics import mean
import os
import csv

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Directorios
dir_test = '../02_Generacion_del_dataset/dataset/test_1280/'
dir_test_mask = '../02_Generacion_del_dataset/dataset/test_mask_1280/'
dir_salida = './output/'
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

tolerancia_color = 30

class punto(NamedTuple):
	x: float
	y: float
class forma(NamedTuple):
	p_image: punto
	p_objeto: punto
	tipo: str
	color: str
	peri: float
	area: float

patron=[]

patron.append(forma(p_image=punto(x=80, y=70), p_objeto=punto(x=0, y=0), tipo='triangulo', color='rojo', peri=0,area =0))
patron.append(forma(p_image=punto(x=217, y=90), p_objeto=punto(x=137, y=20), tipo='cuadrilatero', color='azul', peri=0,area =0))
patron.append(forma(p_image=punto(x=80, y=210), p_objeto=punto(x=0, y=140), tipo='triangulo', color='verde', peri=0,area =0))
patron.append(forma(p_image=punto(x=217, y=215), p_objeto=punto(x=137, y=145), tipo='cuadrilatero', color='rojo', peri=0,area =0))
patron.append(forma(p_image=punto(x=80, y=350), p_objeto=punto(x=0, y=280), tipo='triangulo', color='azul', peri=0,area =0))
patron.append(forma(p_image=punto(x=217, y=340), p_objeto=punto(x=137, y=270), tipo='cuadrilatero', color='verde', peri=0,area =0))

#colores de las formas en RGB
colores_formas = {
    "(145, 105, 2)": 'azul', # 
    "(31, 28, 192)": 'rojo', # 
    "(80, 174, 0)": 'verde'
}

#colores de los fondos
gris_base = (83,84,85)
negro_fondo = (0,0,0)

def drawAxis(img, origen, imgpts):
	
	print(tuple(imgpts[0].ravel()))
	print(origen)
	try:
		img = cv2.line(img, origen, np.array(tuple(imgpts[0].ravel())).astype(int), (255,0,0), 2)
	except:
		print('fallo al dibujar flechita')
	try:
		img = cv2.line(img, origen, np.array(tuple(imgpts[1].ravel())).astype(int), (0,255,0), 2)
	except:
		print('fallo al dibujar flechita')
	try:
		img = cv2.line(img, origen, np.array(tuple(imgpts[2].ravel())).astype(int), (0,0,255), 2)
	except:
		print('fallo al dibujar flechita')
	return img

def estima_pose(img_ori,mask_ori):
	img = cv2.bitwise_and(img_ori, mask_ori)

	#cv2.imshow('img', img)
	#cv2.waitKey(0)

	# Pasamos a coordenadas cromáticas
	r, g, b = cv2.split(img) 

	im_sum = np.sum(img, axis=2)

	rg_chrom_r = np.ma.divide(1.*r, im_sum)
	rg_chrom_g = np.ma.divide(1.*g, im_sum)
	rg_chrom_b = np.ma.divide(1.*b, im_sum)

	rg_chrom = np.zeros_like(img)
	#normalizamos
	rg_chrom[:,:,0] = np.clip(np.uint8(rg_chrom_r*255), 0, 255)
	rg_chrom[:,:,1] = np.clip(np.uint8(rg_chrom_g*255), 0, 255)
	rg_chrom[:,:,2] = np.clip(np.uint8(rg_chrom_b*255), 0, 255)

	#cv2.imshow('img', rg_chrom)
	#cv2.waitKey(0)

	Z = rg_chrom.reshape((-1,3))
	# convertimos a float32
	Z = np.float32(Z)
	# parametrizamos k-means, criterio y número de grupos
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K =5
	compactness,labels,centers=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	# Ahora reducimos los colores de la imagen
	centers = np.uint8(centers)
	print(centers)
	labels = labels.reshape((img.shape[:-1]))
	img_reduced = np.uint8(centers)[labels]
	#cv2.imshow('img_reduced', img_reduced)
	#cv2.waitKey(0)

	kernel_3 = np.ones((3,3),np.uint8)		

	mask = np.zeros(img.shape[:2], np.uint8)	

	# nos quedamos con las zonas que no son ni el negro del fondo de la imagen ni el gris del fondo de la plataforma
	for i, c in enumerate(centers):
		dist_gris = np.linalg.norm(c-gris_base)	#calculamos la distancia euclídea
		dist_negro = np.linalg.norm(c-negro_fondo)
		if dist_gris>tolerancia_color and dist_negro>tolerancia_color:
			mask = cv2.bitwise_or(cv2.inRange(labels, i, i),mask)
			#cv2.imshow('mask',mask)
			#print(color)
			#cv2.waitKey(0)

	#cv2.imshow('mask',mask)
	#cv2.waitKey(0)

	#eliminamos ruido
	mask = cv2.erode(mask,kernel_3,iterations = 1)
	mask = cv2.dilate(mask,kernel_3,iterations = 1)
		
	#cv2.imshow('mask_filtrada',mask)
	#cv2.waitKey(0)

	# buscamos los contornos
	contours, _ = cv2.findContours(
		mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Guardamos los contornos en lista de formas	    
	formas = []
	for contour in contours:

		#contour = cv2.convexHull(contour)	#no mejora

		# cv2.approxPloyDP() aproxima el contorno a un polígono, minizando el número de lados
		# usamos el perímetro del contorno para ajustar la sensibilidad del algoritmo
		peri = cv2.arcLength(contour, True)
		#print(peri)
		approx = cv2.approxPolyDP(
			contour, 0.08 * peri, True)		# este parámetro 0.08*perimetro es clave, habría que 
								# intentar determinarlo de manera automática
		# dibujamos contornos
		cv2.drawContours(img, [approx], 0, (0, 0, 255), 1)

		# buscamos el centroide
		M = cv2.moments(contour)
		x = 0
		y = 0
		if M['m00'] != 0.0:
			x = int(M['m10']/M['m00'])
			y = int(M['m01']/M['m00'])
		
		# forma geométrica
		if len(approx) == 3:
			geom = 'triangulo'
		elif len(approx) == 4:
			geom = 'cuadrilatero'
		elif len(approx) == 5:
			geom = 'pentagono'
		elif len(approx) == 6:
			geom = 'hexagono'
		elif len(approx) > 10:
			geom = 'circulo'
		else:
			geom = 'desconocido'

		color = 'desconocido'

		# asignamos a la forma la categoría de color que tiene más cerca en el espacio RGB
		for c in colores_formas:
			a = rg_chrom[y,x]
			b = np.asarray(eval(c))
			dist = np.linalg.norm(a-b)	#calculamos la distancia euclídea
			if dist<tolerancia_color:
				color = colores_formas[c]

		# guardamos la forma en una lista de formas
		formas.append(forma(punto(x,y),punto(0,0),geom,color,peri,M['m00']))
		#print(forma(punto(x,y),punto(0,0),geom,color,peri,M['m00']))
		#cv2.imshow('contorno', img)
		#cv2.waitKey(0)
	print('##################\n')	
	#print(formas)
	#cv2.imshow('contorno', img)
	#cv2.waitKey(0)
	#cv2.imwrite('./output/05_contornos.jpg',img)
		
	#filtramos por tipo (solo triangulos o cuadriláteros)	
	filtrado_iter = filter(lambda formas: ((formas.tipo=='triangulo')or(formas.tipo=='cuadrilatero')) , formas)	
	formas = list(filtrado_iter)
	#quitamos las de colores desconocidos	
	filtrado_iter = filter(lambda formas: (formas.color=='verde')or(formas.color=='rojo')or(formas.color=='azul'), formas)
	formas = list(filtrado_iter)

	# nos quedamos con las que coincidan en color y forma con el patrón,
	# y estas serán nuestras referencias para el PnP
	referencias = []
	for f in formas:
		for p in patron:
			if ((f.tipo==p.tipo) and (f.color==p.color)):
				f = f._replace(p_objeto=p.p_objeto)
				referencias.append(f)
				img = cv2.circle(img, (f.p_image.x,f.p_image.y), radius=2, color=(0, 0, 2), thickness=-1)
				
	print('##################\n')	
	print(referencias)

	# Construimos las listas de puntos objeto e imagen
	objectPoints = []
	imagePoints = []
	
	for f in referencias:
		objectPoints.append([f.p_objeto.x,f.p_objeto.y,0.])
		imagePoints.append([f.p_image.x,f.p_image.y])

	print(np.array(objectPoints).astype(np.float32))
	print(imagePoints)
	
	# cv::solvePnP() no funciona con menos de 4 puntos, y de nuestra imagen base solo extraemos 6 puntos
	# por lo que limitamos la aplicación del algoritmo a que tengamos entre 4 y 6 puntos
	if len(np.array(imagePoints).astype(np.float32))<4 or len(np.array(imagePoints).astype(np.float32))>6:
		ret = False
		return ret, _, _, _
	   
	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)*40 #ejes

	#1280x720
	mtx = np.float32([[962., 0., 667.],
			    [0., 964., 339.],
			    [0., 0.,   1.]])
	dist = np.float32([ 0.12625537, -0.46566645,  0.00204414,  0.00085982,  0.46032687])	#1280x720
	
	#640x480
	#mtx = np.float32([[628.12993914,   0. ,        334.86597715],
	# [  0.,         629.01615969, 243.96642583],
	# [  0.,           0.,           1.        ]])	    
	#dist = np.float32([ 0.05082377, -0.07573206,  0.01024431, -0.00179136, -0.43260539])		#640x480
	
	try:
		ret,rvecs,tvecs = cv2.solvePnP(np.array(objectPoints).astype(np.float32), np.array(imagePoints).astype(np.float32), mtx, dist)
	except:
		ret = False
		return ret, _, _, _   

	# project 3D points to image plane
	imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
	print('*+*+*+*+*+*+*+*+')
	#print(imgpts)
	origen = punto(x=0, y=0)
	for f in formas:
		# el origen es el triángulo rojo, si no se ha encontrado esta referencia no se va a dibujar bien,
		# esta limitación se podría quitar en un futuro, dibujando los ejes en cualquier otra forma
		# o en una esquina de la imagen base
		if ((f.tipo=='triangulo')and(f.color=='rojo')):
			origen = f.p_image
	img = drawAxis(img_ori,tuple(origen),imgpts)

	#cv2.imshow('shapes', img)
	#cv2.imwrite('./output/06_pose.jpg',img)
	
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	ret = True
	return ret, img, rvecs, tvecs

#generamos el fichero de anotaciones de poses
anotation = open(dir_salida+'poses_estimadas.txt', 'w')
writer = csv.writer(anotation)
header = ['imagen','rotacion','traslacion']
writer.writerow(header)
	
for image in sorted(os.listdir(dir_test)):
	print(image)
	#imagen
	img_ori = cv2.imread(dir_test+image)
	#máscara
	mask_ori = cv2.imread(dir_test_mask+image)
	#estimamos pose
	ret, img_final, R, T = estima_pose(img_ori, mask_ori)
	
	if ret:
		row = [image, R, T]
		writer.writerow(row)
		
		cv2.imwrite(dir_salida+image,img_final)
	
	
anotation.close()

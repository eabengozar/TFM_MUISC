#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import NamedTuple
from statistics import mean
import rospy
#from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
#from collections import namedtuple
#from typing import NamedTuple

import NotCvBridge as bridge

tolerancia_color = 50

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


category_colors = {
    "(130, 99, 16)": 'azul', # 
    "(41, 55, 150)": 'rojo', # 
    "(28, 45, 179)": 'rojo',
    "(45, 151, 56)": 'verde',
    "(71, 118, 62)": 'verde'
}

gris_base = (83,84,85)
negro_fondo = (0,0,0)

#bridge = CvBridge()
global cv2_imagen_entrada, cv2_mask_entrada

def drawAxis(img, origen, imgpts):
	#corner = tuple(corners[0].ravel())
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

def estima_pose(img_ori,mask_ori,size):
	img = cv2.bitwise_and(img_ori, mask_ori)

	#cv2.imshow('img', img)
	#cv2.waitKey(0)

	r, g, b = cv2.split(img) 

	im_sum = np.sum(img, axis=2)

	rg_chrom_r = np.ma.divide(1.*r, im_sum)
	rg_chrom_g = np.ma.divide(1.*g, im_sum)
	rg_chrom_b = np.ma.divide(1.*b, im_sum)

	# Visualize rg Chromaticity --> DEBUGGING
	rg_chrom = np.zeros_like(img)

	rg_chrom[:,:,0] = np.clip(np.uint8(rg_chrom_r*255), 0, 255)
	rg_chrom[:,:,1] = np.clip(np.uint8(rg_chrom_g*255), 0, 255)
	rg_chrom[:,:,2] = np.clip(np.uint8(rg_chrom_b*255), 0, 255)

	#cv2.imshow('img', rg_chrom)
	#cv2.waitKey(0)


	#mask = np.zeros(img.shape[:2], np.uint8)
	#mask = cv2.bitwise_or(cv2.inRange(rg_chrom, azul_min, azul_max),mask)
	#mask = cv2.bitwise_or(cv2.inRange(rg_chrom, verde_min, verde_max),mask)
	#mask = cv2.bitwise_or(cv2.inRange(rg_chrom, rojo_min, rojo_max),mask)

	Z = rg_chrom.reshape((-1,3))
	# convert to np.float32
	Z = np.float32(Z)
	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K =15
	compactness,labels,centers=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	# Ahora reducimos los colores de la imagen
	centers = np.uint8(centers)
	print(centers)
	labels = labels.reshape((img.shape[:-1]))
	img_reduced = np.uint8(centers)[labels]
	#cv2.imshow('img_reduced', img_reduced)
	#cv2.waitKey(0)

	kernel_3 = np.ones((3,3),np.uint8)		#OJO CON EL TAMAÑO DEL KERNEL EN FUNCIÓN DEL TAMAÑO DE LA IMAGEN
	kernel_5 = np.ones((5,5),np.uint8)

	mask = np.zeros(img.shape[:2], np.uint8)	

	for i, c in enumerate(centers):
		#b = np.asarray(eval(gris_base))
		dist_gris = np.linalg.norm(c-gris_base)	#calculamos la distancia euclídea
		#b = np.asarray(eval(negro_fondo))
		dist_negro = np.linalg.norm(c-negro_fondo)
		if dist_gris>tolerancia_color and dist_negro>tolerancia_color:
			mask = cv2.bitwise_or(cv2.inRange(labels, i, i),mask)
			#cv2.imshow('mask',mask)
			#print(color)
			#cv2.waitKey(0)

	#cv2.imshow('mask',mask)
	#cv2.waitKey(0)


	#mask = cv2.dilate(mask,kernel_3,iterations = 3)
	#mask = cv2.erode(mask,kernel_3,iterations = 3)

	mask = cv2.erode(mask,kernel_3,iterations = 1)
	mask = cv2.dilate(mask,kernel_3,iterations = 1)
		
	#cv2.imshow('mask_filtrada',mask)
	#cv2.waitKey(0)

	# using a findContours() function
	contours, _ = cv2.findContours(
		mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	i = 0

	# Guardamos los contornos en nuestra lista de formas	    
	formas = []
	for contour in contours:

		#contour = cv2.convexHull(contour)	#NO SÉ SI MEJORA

		# cv2.approxPloyDP() function to approximate the shape
		peri = cv2.arcLength(contour, True)
		#print(peri)
		approx = cv2.approxPolyDP(
			#contour, 0.01 * peri, True)		#CREO QUE AQUÍ SE DECIDE CÓMO DE FINO ES EL AJUSTE, NOS INTERESA BRUTO?
			contour, 0.08 * peri, True)		#CREO QUE AQUÍ SE DECIDE CÓMO DE FINO ES EL AJUSTE, NOS INTERESA BRUTO? 0.04

		# using drawContours() function
		cv2.drawContours(img, [approx], 0, (0, 0, 255), 1)

		# finding center point of shape
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

		for c in category_colors:
			a = rg_chrom[y,x]
			b = np.asarray(eval(c))
			dist = np.linalg.norm(a-b)	#calculamos la distancia euclídea
			if dist<tolerancia_color:
				color = category_colors[c]

		formas.append(forma(punto(x,y),punto(0,0),geom,color,peri,M['m00']))
		#print(forma(punto(x,y),punto(0,0),geom,color,peri,M['m00']))
		#print(approx)
		#cv2.imshow('contorno', img)
		#cv2.waitKey(0)
	print('##################\n')	
	#print(formas)
	#cv2.imshow('contorno', img)
	#cv2.waitKey(0)
	#cv2.imwrite('./output/05_contornos.jpg',img)
		
	#filtramos por tipo (solo círculos o cuadriláteros)	
	filtrado_iter = filter(lambda formas: ((formas.tipo=='triangulo')or(formas.tipo=='cuadrilatero')) , formas)	
	formas = list(filtrado_iter)
	#quitamos las de colores desconocidos
	#filtrado_iter = filter(lambda formasimport csv: (formas.color!='desconocido'), formas)	
	filtrado_iter = filter(lambda formas: (formas.color=='verde')or(formas.color=='rojo')or(formas.color=='azul'), formas)
	formas = list(filtrado_iter)

	#nos quedamos con las que coincidan en color y forma con el patrón de referencia
	referencias = []
	for f in formas:
		for p in patron:
			if ((f.tipo==p.tipo) and (f.color==p.color)):
				#items[node.ind] = items[node.ind]._replace(v=node.v)
				f = f._replace(p_objeto=p.p_objeto)
				referencias.append(f)
				img = cv2.circle(img, (f.p_image.x,f.p_image.y), radius=2, color=(0, 0, 2), thickness=-1)
				
	print('##################\n')	
	print(referencias)

#	objectPoints = np.zeros((6, 3))	#OJO SÚPER IMPORTANTE, NO PUEDE HABER DATOS EN LOS VECTORES COMO CEROS, NO FUNCIONA BIEN EL SOLVE
#	imagePoints = np.zeros((6, 2))
#	i=0
#	for f in referencias:
#		objectPoints[i,0] = f.p_objeto.x
#		objectPoints[i,1] = f.p_objeto.y
#		imagePoints[i,0] = f.p_image.x
#		imagePoints[i,1] = f.p_image.y
#		i += 1
#		if i==6:	#chapucilla
#			break

	objectPoints = []
	imagePoints = []
	
	for f in referencias:
		objectPoints.append([f.p_objeto.x,f.p_objeto.y,0.])
		imagePoints.append([f.p_image.x,f.p_image.y])

	print(np.array(objectPoints).astype(np.float32))
	print(imagePoints)
	
	if len(np.array(imagePoints).astype(np.float32))<4 or len(np.array(imagePoints).astype(np.float32))>6:
		ret = False
		return ret, _, _, _
	   
	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)*40 #ejes de 20mm??

	if size=='1280x720':
		#1280x720
		mtx = np.float32([[962., 0., 667.],
				    [0., 964., 339.],
				    [0., 0.,   1.]])
		dist = np.float32([ 0.12625537, -0.46566645,  0.00204414,  0.00085982,  0.46032687])	#1280x720
	elif size=='640x480':
		#640x480
		mtx = np.float32([[628.12993914,   0. ,        334.86597715],
		 [  0.,         629.01615969, 243.96642583],
		 [  0.,           0.,           1.        ]])	    
		dist = np.float32([ 0.05082377, -0.07573206,  0.01024431, -0.00179136, -0.43260539])		#640x480
	
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
		if ((f.tipo=='triangulo')and(f.color=='rojo')):
			origen = f.p_image
	img = drawAxis(img_ori,tuple(origen),imgpts)

	# displaying the image after drawing contours
	#cv2.imshow('shapes', img)

	#cv2.imwrite('./output/06_pose.jpg',img)

	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	ret = True
	return ret, img, rvecs, tvecs

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# callback de imagen
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def image_callback(msg):
	#print("Received an image!")
	global cv2_imagen_entrada
	try:
		# Convert your ROS Image message to OpenCV2
		#cv2_imagen_entrada = bridge.imgmsg_to_cv2(msg, "rgb8")
		cv2_imagen_entrada = bridge.imgmsg_to_cv2(msg)
	except:# CvBridgeError as e:
		#print(e)
		rospy.loginfo('error callback_Imagen')
	else:
		rospy.loginfo("callback_Imagen")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# callback de la máscara
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mask_callback(msg):
	#print("Received an image!")
	try:
		# Convert your ROS Image message to OpenCV2
		#cv2_mask_entrada = bridge.imgmsg_to_cv2(msg, "rgb8")
		cv2_mask_entrada = bridge.imgmsg_to_cv2(msg)
	except:# CvBridgeError as e:
		#print(e)
		rospy.loginfo('error callback_Mask')
	else:
		rospy.loginfo("callback_Mask")
		ret, img_final, R, T = estima_pose(cv2_imagen_entrada, cv2_mask_entrada,'1280x720')
		pose_pub.publish(bridge.cv2_to_imgmsg(img_final))


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Principal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
		
	#inicializamos nodo
	rospy.init_node('estima_pose')

	#suscripción para la imagen de la cámara
	camara_sub = rospy.Subscriber("/image_fotos", Image, image_callback)
	#suscripción para la máscara
	camara_sub = rospy.Subscriber("/object_detection/output_mask", Image, mask_callback)

	pose_pub = rospy.Publisher("/estima_pose/image_pose", Image, queue_size=1)

	rospy.spin()


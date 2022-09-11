import numpy as np
import cv2 as cv
import glob

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Configuración del directorio de las imágenes de calibración
dir_images = './Imgs_Calibracion/1280x720/'
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# criterio de terminación de la búsqueda fina de esquinas
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# preparamos los puntos objeto en función del tablero que usamos, (0,0,0), (1,0,0), (2,0,0) ....,(7,11,0)
objp = np.zeros((7*11,3), np.float32)
objp[:,:2] = np.mgrid[0:11,0:7].T.reshape(-1,2)*20 #cuadrados de 20mm PONIENDO *20 DEVUELVE LOS TRECS EN DISTANCIA REAL, NO EN PÍXELES

# Arrays de los puntos objeto y los puntos imagen
objpoints = [] # puntos 3d en el mundo real (mm)
imgpoints = [] # puntos 2d en el plano de la imagen (px)

images = glob.glob(dir_images + '*.jpg')
for fname in images:
	img = cv.imread(fname)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# Búsqueda de las esquinas del tablero
	ret, corners = cv.findChessboardCorners(gray, (11,7), None)
	# si se encuentran, se añaden a las lista de puntos objeto y puntos imagen, una vez afinados
	if ret == True:
		objpoints.append(objp)
		corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners)
		# Dibujamos las esquinas encontradas
		cv.drawChessboardCorners(img, (11,7), corners2, ret)
		cv.imshow('img', img)
		cv.waitKey(500)
cv.destroyAllWindows()

#Llamada al método de calibración
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('++++++++++++++++++++++')
print('+Matriz Intrinseca+')
print(mtx)
print('+Coef. de Distorsion+')
print(dist)
print('++++++++++++++++++++++')

#Cálculo del error de reproyección
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "Error medio de reproyección: {}".format(mean_error/len(objpoints)) )

import numpy as np
import cv2 as cv
import glob

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Configuración de la resolución a testear
resolucion = '640x480' #'1280x720'
# Configuración del directorio de las imágenes de test
dir_images = './test/'+resolucion+'/'
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

# Matriz intrínseca y coeficientes de distorsión según resolución
if resolucion=='1280x720':
	#1280x720
	mtx = np.float32([[962.16316413,   0.,         667.5837451 ],
 	 [  0.,         963.60328012, 339.02306189],
	 [  0.,           0.,           1.        ]])
	dist = np.float32([ 0.12625537, -0.46566645,  0.00204414,  0.00085982,  0.46032687])	#1280x720
elif resolucion=='640x480':
	#640x480
	mtx = np.float32([[628.12993914,   0. ,        334.86597715],
	 [  0.,         629.01615969, 243.96642583],
	 [  0.,           0.,           1.        ]])	    
	dist = np.float32([ 0.05082377, -0.07573206,  0.01024431, -0.00179136, -0.43260539])		#640x480
		

# llamámos al método para resolver la perspectiva desde N puntos
rvecs = []
tvecs = []
for i in range(len(objpoints)):
	ret,rr,tt = cv.solvePnP(objpoints[i], imgpoints[i], mtx, dist)
	rvecs.append(rr)
	tvecs.append(tt)

# Cálculo del error de reproyección
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "Error medio de reproyección: {}".format(mean_error/len(objpoints)) )

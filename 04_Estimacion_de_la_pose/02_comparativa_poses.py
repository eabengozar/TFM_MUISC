import numpy as np
import os
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
import cv2
from numpy.linalg import inv

import matplotlib.pyplot as plt

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ficheros de poses reales, estimadas y de salida para errores
file_poses = '../02_Generacion_del_dataset/dataset/test_1280/poses.txt'
file_estimaciones = './output/poses_estimadas.txt'
file_errores = './output/errores.txt'
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

f_poses = open(file_poses)
f_estimaciones = open(file_estimaciones)

csv_reader_poses = csv.DictReader(f_poses)
csv_reader_estimaciones = csv.DictReader(f_estimaciones)

poses = list(csv_reader_poses)
estimaciones = list(csv_reader_estimaciones)

mse_rotacion = []
mse_traslacion = []
varscore_rot = []
varscore_tras = []

rot = []
pose_tvec = []
estimacion_rotVec = []
estimacion_tvec = []
error_norm_t = []
error_norm_r = []
error_norm = []

mean_error_t = 0
mean_error_r = 0
mean_error = 0

for i in range(len(poses)):
	for j in range(len(estimaciones)):
		if poses[i]['imagen']==estimaciones[j]['imagen']:
			#Rotación real	
			rot.append([i,np.matrix(poses[i]['rotacion']).reshape((4,4))[0:3,0:3]]) #vector 3x3
			pose_rotVec,_ = cv2.Rodrigues(rot[-1][1]) #vector 1x3
			#Traslación real
			pose_tvec_m = np.matrix(poses[i]['traslacion']).reshape((4,4))
			pose_tvec.append([i,pose_tvec_m[0:3,3:4]])
			#Rotación estimada
			estimacion_rot = np.matrix(estimaciones[j]['rotacion']) #vector 3x1
			estimacion_rotVec_,_ = cv2.Rodrigues(estimacion_rot.T) #vector 3x3
			estimacion_rotVec.append([i,estimacion_rotVec_])
			#Traslación estimada
			estimacion_tvec.append([i,np.matrix(estimaciones[j]['traslacion']).T])
			print('++++++++++++++++++++')
			print('++++++++++++++++++++')
			#normas de traslación, rotación y producto de ambas
			error_norm_t.append(cv2.norm(pose_tvec[-1][1].astype(np.float32), estimacion_tvec[j][-1].astype(np.float32), cv2.NORM_L2)/len(estimaciones))
			error_norm_r.append(cv2.norm(rot[-1][1].astype(np.float32), estimacion_rotVec[j][-1].astype(np.float32), cv2.NORM_L2)/len(estimaciones))
			error_norm.append(cv2.norm(rot[-1][1].astype(np.float32)*pose_tvec[-1][1].astype(np.float32), estimacion_rotVec[j][-1].astype(np.float32)*estimacion_tvec[j][-1].astype(np.float32), cv2.NORM_L2)/len(estimaciones))
			print('error_norm_t: '+str(i)+'-> '+str(error_norm_t[i]))
			print('error_norm_r: '+str(i)+'-> '+str(error_norm_r[i]))
			print('error_norm: '+str(i)+'-> '+str(error_norm[i]))
			mean_error_t += error_norm_t[i]
			mean_error_r += error_norm_r[i]
			mean_error += error_norm[i]	
	# si no hay estimación añadimos el valor de -1 a los errores
	if (len(error_norm_t)-1)!=i:
		error_norm_t.append(-1)
		error_norm_r.append(-1)	
		error_norm.append(-1)	


print('++++++++++++++++++++')
#errores medios
print('mean_error_t: '+ str(mean_error_t/len(estimacion_tvec)))
print('mean_error_r: '+ str(mean_error_r/len(estimacion_tvec)))
print('mean_error: '+ str(mean_error/len(estimacion_tvec)))
#graficamos los errores
x = np.arange(len(error_norm_t))
plt.bar(x,error_norm_t,align='center')
plt.xticks(np.arange(min(x), max(x)+1, 2.0))
plt.xlabel('Nº de imagen de Test')
plt.ylabel('Error de traslación')
plt.show()
plt.bar(x,error_norm_r,align='center')
plt.xticks(np.arange(min(x), max(x)+1, 2.0))
plt.xlabel('Nº de imagen de Test')
plt.ylabel('Error de rotacion')
plt.show()
plt.bar(x,error_norm,align='center')
plt.xticks(np.arange(min(x), max(x)+1, 2.0))
plt.xlabel('Nº de imagen de Test')
plt.ylabel('Error de (rotacion x traslacion)')
plt.show()

#generamos el fichero de errores de poses
anotation = open(file_errores, 'w')
writer = csv.writer(anotation)
header = ['imagen','error_traslacion','error_rotacion','error_rotacion_x_traslacion']
writer.writerow(header)

for i in range(len(error_norm_t)):
	row = ['imagen_'+str(i), error_norm_t[i], error_norm_r[i],error_norm[i]]
	writer.writerow(row)
	
anotation.close()

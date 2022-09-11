import glob
import cv2

from image_to_coco_json_converter.src.create_annotations import *

# Ids de las categorías del dataset
category_ids = {
    "imagen_base": 0
}

# Define los colores correspondientes a cada categoría
category_colors = {
	"(255, 255, 255)": 0, # Máscara de la Imagen_base
	"(0, 0, 0)": 1 # fondo
}


# Categorías que son multipolígono
multipolygon_ids = [0,1] #[2, 5, 6]

# distancia máxima permitida para pertenecer a una categoría
tolerancia_color = 50

# Creamos las "images" y "annotations"
def images_annotations_info(maskpath):
	annotation_id = 0
	image_id = 0
	annotations = []
	images = []

	for mask_image in sorted(glob.glob(maskpath + "*.jpg")):
	        
		original_file_name = os.path.basename(mask_image).split(".")[0] + ".jpg"
		
		print(original_file_name)

		# Abrimos la imagen y nos aseguramos y la convertimos en RGB para asegurarnos
		mask_image_open = Image.open(mask_image).convert("RGB")
		#mask_image_open.show()       
		w, h = mask_image_open.size

		#+++++++++++++++++++++++++++++++++++++++
		# Reducimos los colores segmentando por k-means (para evitar grises en bordes)
		#+++++++++++++++++++++++++++++++++++++++
		open_cv_image = np.array(mask_image_open) 
		Z = open_cv_image.reshape((-1,3))
		# convertimos a float32
		Z = np.float32(Z)
		# parametrizamos k-means, criterio y número de grupos
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		K =len(category_colors)
		compactness,labels,centers=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
		centers = np.uint8(centers)
		labels = labels.reshape((open_cv_image.shape[:-1]))
		reduced = np.uint8(centers)[labels]
		im_reduced = Image.fromarray(reduced)
		
		# creamos la anotación "images"
		image = create_image_annotation(original_file_name, w, h, image_id)
		images.append(image)

		#ahora comprobaremos las diferentes categorías que existen en la image
		sub_masks = create_sub_masks(im_reduced, w, h)
		for color, sub_mask in sub_masks.items():
		
			#+++++++++++++++++++++++++++++++++++++++
			# Asignamos el color de la categoría que  
			# esté por debajo de cierta tolerancia
			#+++++++++++++++++++++++++++++++++++++++		
			for c in category_colors:
				a = np.asarray(eval(c))
				#print(a)
				b = np.asarray(eval(color))
				dist = np.linalg.norm(a-b)
				#print(dist)
				if dist<tolerancia_color:
					color = c
					#print(color)
					
			category_id = category_colors[color]	
			print(category_id)
			
			if category_id in category_ids.values():					
				# información necesaria para "annotations"
				polygons, segmentations = create_sub_mask_annotation(sub_mask)

				# si es multipolígono o no
				if category_id in multipolygon_ids:
					# se combinan los polígonos del multipolígono
					multi_poly = MultiPolygon(polygons)
							
					annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

					annotations.append(annotation)
					annotation_id += 1
				else:
					for i in range(len(polygons)):
						segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]

						annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)

						annotations.append(annotation)
						annotation_id += 1
		image_id += 1
	return images, annotations, annotation_id

if __name__ == "__main__":
    # Se carga la estructura del formato json de COCO
    coco_format = get_coco_json_format()
    
    # lo haremos para los distintos conjuntos
    for keyword in ["train", "val","test"]:
        mask_path = "dataset/{}_mask/".format(keyword)
        
        # anotación "categories"
        coco_format["categories"] = create_category_annotation(category_ids)   
    
        # anotaciones "images" y "annotations"
        coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

        with open("output_json/{}.json".format(keyword),"w") as outfile:
            json.dump(coco_format, outfile)
        
        print("Creadas %d anotaciones para las imágenes del directorio: %s" % (annotation_cnt, mask_path))


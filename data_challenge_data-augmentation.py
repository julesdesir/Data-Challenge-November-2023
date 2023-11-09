import cv2
import numpy as np
import os
import pandas as pd
from google.colab import drive #j'ai travaillé sur google colab pour ne pas être en local
import random as rd
import shutil

# Monter Google Drive
drive.mount('/content/drive')


# CREATION DES DOSSIERS

# Dossiers et fichiers d'entrée et de sortie
dataset_folder = '/content/drive/My Drive/data_challenge_mines/datasets_train/'
newdataset_folder = '/content/drive/My Drive/data_challenge_mines_v2/data_challenge_mines/datasets_train/augmented_datasets'
train_folder = os.path.join(dataset_folder, 'train')
annotation_file = os.path.join(
    dataset_folder, 'train_annotation/_annotation.csv')
output_folder = os.path.join(newdataset_folder, 'augmented_data_train')
output_annotation_file = os.path.join(
    newdataset_folder, 'augmented_data_annotation.csv')

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(newdataset_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# PREPARATION DE NOTRE DATASET AUGMENTE : IMPORTE LE PREEXISTANT
# Copier toutes les images de train vers augmented_data_train
for file in os.listdir(train_folder):
    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
        source_path = os.path.join(train_folder, file)
        destination_path = os.path.join(output_folder, file)
        shutil.copy(source_path, destination_path)

# Copier le contenu de _annotation.csv vers augmented_data_annotation.csv
shutil.copy(annotation_file, output_annotation_file)


# ROTATION et REFLEXION

# Fonction pour appliquer la transformation et adapter le cadre
def apply_transformation(image, x_min, y_min, x_max, y_max, transformation_type):
    # Appliquer la transformation
    if transformation_type == 'rotated_left':
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif transformation_type == 'rotated_right':
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif transformation_type == 'flipped_horizontal':
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    elif transformation_type == 'flipped_vertical':
        rotated_image = cv2.flip(image, 1)

    hauteur, largeur, canaux = image.shape
    # Ajuster le cadre pour la transformation
    if transformation_type == 'rotated_right':
        x_min_rotated = hauteur-y_max
        y_min_rotated = x_min
        x_max_rotated = hauteur - y_min
        y_max_rotated = x_max

    elif transformation_type == 'rotated_left':
        x_min_rotated = y_min
        y_min_rotated = largeur - x_max
        x_max_rotated = y_max
        y_max_rotated = largeur - x_min

    elif transformation_type == 'flipped_horizontal':
        x_min_rotated = largeur - x_max
        y_min_rotated = hauteur - y_max
        x_max_rotated = largeur - x_min
        y_max_rotated = hauteur - y_min

    elif transformation_type == 'flipped_vertical':
        x_min_rotated = largeur - x_max
        y_min_rotated = y_min
        x_max_rotated = largeur - x_min
        y_max_rotated = y_max

    return rotated_image, x_min_rotated, y_min_rotated, x_max_rotated, y_max_rotated


# Charger les annotations à partir du fichier CSV
annotations = pd.read_csv(annotation_file)
new_df = pd.DataFrame()


# EDITION DES ROTATIONS

# Appliquer des transformations à chaque image
for index, row in annotations.iterrows():
    image_name = row['im_name']
    x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
    class_label = row['class']
    model = row['models']

    # Charger l'image
    image = cv2.imread(os.path.join(train_folder, image_name))

    # Appliquer la ROTATION A GAUCHE
    rotated_left_image, x_min_rotated_left, y_min_rotated_left, x_max_rotated_left, y_max_rotated_left = apply_transformation(
        image, x_min, y_min, x_max, y_max, 'rotated_left')
    rotated_left_image_name = f"rotated_left_{image_name}"
    cv2.imwrite(os.path.join(output_folder,
                rotated_left_image_name), rotated_left_image)

    # Ajouter une ligne au fichier CSV de sortie pour la transformation de gauche
    output_annotation_left = pd.DataFrame({
        'im_name': [rotated_left_image_name],
        'x_min': [x_min_rotated_left],
        'y_min': [y_min_rotated_left],
        'x_max': [x_max_rotated_left],
        'y_max': [y_max_rotated_left],
        'class': [class_label],
        'models': [model]
    })

    new_df = pd.concat([new_df, output_annotation_left], axis=0)

    # Appliquer la ROTATION A DROITE
    rotated_right_image, x_min_rotated_right, y_min_rotated_right, x_max_rotated_right, y_max_rotated_right = apply_transformation(
        image, x_min, y_min, x_max, y_max, 'rotated_right')
    rotated_right_image_name = f"rotated_right_{image_name}"
    cv2.imwrite(os.path.join(output_folder,
                rotated_right_image_name), rotated_right_image)

    # Ajouter une ligne au fichier CSV de sortie pour la transformation de droite
    output_annotation_right = pd.DataFrame({
        'im_name': [rotated_right_image_name],
        'x_min': [x_min_rotated_right],
        'y_min': [y_min_rotated_right],
        'x_max': [x_max_rotated_right],
        'y_max': [y_max_rotated_right],
        'class': [class_label],
        'models': [model]
    })

    new_df = pd.concat([new_df, output_annotation_right], axis=0)

    # Appliquer la REFLEXION HORIZONTALE(rotation 180°)
    flipped_horizontal_image, x_min_flipped_horizontal, y_min_flipped_horizontal, x_max_flipped_horizontal, y_max_flipped_horizontal = apply_transformation(
        image, x_min, y_min, x_max, y_max, 'flipped_horizontal')
    flipped_horizontal_image_name = f"flipped_horizontal_{image_name}"
    cv2.imwrite(os.path.join(output_folder,
                flipped_horizontal_image_name), flipped_horizontal_image)

    # Ajouter une ligne au fichier CSV de sortie pour la réflexion horizontale
    output_annotation_flipped_horizontal = pd.DataFrame({
        'im_name': [flipped_horizontal_image_name],
        'x_min': [x_min_flipped_horizontal],
        'y_min': [y_min_flipped_horizontal],
        'x_max': [x_max_flipped_horizontal],
        'y_max': [y_max_flipped_horizontal],
        'class': [class_label],
        'models': [model]
    })

    new_df = pd.concat([new_df, output_annotation_flipped_horizontal], axis=0)

    # Appliquer la REFLEXION VERTICALE
    flipped_vertical_image, x_min_flipped_vertical, y_min_flipped_vertical, x_max_flipped_vertical, y_max_flipped_vertical = apply_transformation(
        image, x_min, y_min, x_max, y_max, 'flipped_vertical')
    flipped_vertical_image_name = f"flipped_vertical_{image_name}"
    cv2.imwrite(os.path.join(output_folder,
                flipped_vertical_image_name), flipped_vertical_image)

    # Ajouter une ligne au fichier CSV de sortie pour la réflexion verticale
    output_annotation_flipped_vertical = pd.DataFrame({
        'im_name': [flipped_vertical_image_name],
        'x_min': [x_min_flipped_vertical],
        'y_min': [y_min_flipped_vertical],
        'x_max': [x_max_flipped_vertical],
        'y_max': [y_max_flipped_vertical],
        'class': [class_label],
        'models': [model]
    })

    new_df = pd.concat([new_df, output_annotation_flipped_vertical], axis=0)

existing_df = pd.read_csv(output_annotation_file)
df_final = pd.concat([existing_df, new_df], axis=0)
df_final.to_csv(output_annotation_file, index=False)

# COULEUR et ZOOM sur dataset augmenté déja edité :
# 1 )on prend une image du dataset initial
# 2 )on tire au sort une transformation rotation/reflexion
# 3 )on prend l'image correspondante du dataset augmenté pour lui rajouter une transformation couleur ou zoom

# Charger les annotations à partir du fichier CSV
annotations = pd.read_csv(annotation_file)
augmented_annotations = pd.read_csv(output_annotation_file)


def transformation(a):
    if a == 0:
        return ""
    elif a == 4:
        return "flipped_vertical_"
    elif a == 3:
        return "flipped_horizontal_"
    elif a == 2:
        return "rotated_right_"
    else:
        return "rotated_left_"


i = 0


# Appliquer des transformations à chaque image
for index, row in annotations.iterrows():
    image_name = row['im_name']

    # Images du dataset augmenté
    nb1, nb2 = rd.randint(0, 5), rd.randint(0, 5)
    transfo1, transfo2 = transformation(nb1), transformation(nb2)
    image1_name, image2_name = transfo1+image_name, transfo2+image_name

    # Charger l'image
    image1 = cv2.imread(os.path.join(output_folder, image1_name))
    image2 = cv2.imread(os.path.join(output_folder, image2_name))

    # AUGMENTATION COULEUR de l'image 1

    # Définissez la variation aléatoire de couleur

    def color(image):
        # On décompose nos canaux RGB
        IMAGE = []
        for el in cv2.split(image):
            el = el + np.array(rd.randint(0, 100)-50)
            IMAGE.append(el)
        im = cv2.merge(tuple(IMAGE))

        # Assurez-vous que les valeurs restent dans la plage 0-255
        result = np.clip(im, 0, 255).astype(np.uint8)
        return result

    colored_image1 = color(image1)
    # On enregistre la nouvelle image
    cv2.imwrite(os.path.join(output_folder, image1_name), colored_image1)

    # ZOOM de l'image 2
    # On va prendre 4 sommets de la nouvelle image, chacun entre le sommet inital et la boite intiale

    # Information de l'image 2
    row2 = augmented_annotations.loc[augmented_annotations['im_name'] == image2_name]

    # boite initiale
    x_min_2, y_min_2, x_max_2, y_max_2 = row2['x_min'], row2['y_min'], row2['x_max'], row2['y_max']

    # dimension image2
    hauteur2, largeur2, canaux = (image2).shape

    # Nouvelle taille image2 zoomée : nouveau sommet de l'image
    x1 = float(rd.randint(0, int(x_min_2)))
    y1 = float(rd.randint(0, int(y_min_2)))
    x2 = float(rd.randint(0, int(largeur2 - x_max_2)))+x_max_2
    y2 = float(rd.randint(0, int(hauteur2 - y_max_2)))+y_max_2

    image2_zoom = image2[int(y1):int(y2), int(x1):int(x2)]

    # Enregistrez l'image zoomée sous le nom de l'ancienne
    cv2.imwrite(os.path.join(output_folder, image2_name), image2_zoom)

    # actualiser la boite pour l'image zoomée
    row2['x_min'], row2['y_min'], row2['x_max'], row2['y_max'] = x_min_2 - \
        x1, y_min_2-y1, x_max_2-x1, y_max_2-y1

from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
import os

# Load an already trained model
model = YOLO('/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/runs/detect/train6/weights/best.pt')  # load a pretrained model (recommended for training)

# Prediction
results_predict = model.predict(data="/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/local_env/config.yaml", source="/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/datasets_test/test", max_det=1, save=True, save_txt=True) # Only 1 detection per image


# Save the predictions in a csv file
with open('predicted_results_V2.csv', 'w') as csv_file:
    csv_file.write('image_id, predicted_class, xmin, ymin, xmax,ymax')
    csv_file.write('\n')

    # Absolute paths to folders
    folder_predicted_images = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/runs/detect/predict36'
    folder_predicted_labels = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/runs/detect/predict36/labels'

    # All the predictions
    predicted_images = [image for image in os.listdir(folder_predicted_images)]
    predicted_labels = [label for label in os.listdir(folder_predicted_labels)]

    for label in predicted_labels:
        # Image id
        csv_file.write(str(label[:-4]))
        csv_file.write(',')

        # Detection
        path_source_label = os.path.join(folder_predicted_labels, label)
        with open(path_source_label, 'r') as txt_file:
            line = txt_file.readline()
            #print(line)
            length = len(line)
            classes, xmin, ymin, xmax, ymax = '', '', '', '', ''
            for index in range(length):
                if line[index] != ' ':
                    classes = classes + line[index]
                else:
                    last_index = index
                    break
            for index in range(last_index+1, length):
                if line[index] != ' ':
                    xmin = xmin + line[index]
                else:
                    last_index = index
                    break
            for index in range(last_index+1, length):
                if line[index] != ' ':
                    ymin = ymin + line[index]
                else:
                    last_index = index
                    break
            for index in range(last_index+1, length):
                if line[index] != ' ':
                    xmax = xmax + line[index]
                else:
                    last_index = index
                    break
            for index in range(last_index+1, length):
                if line[index] != ' ':
                    ymax = ymax + line[index]
                else:
                    last_index = index
                    break
            
            # Write in the .csv file
            csv_file.write(classes)
            csv_file.write(',')
            csv_file.write(xmin)
            csv_file.write(',')
            csv_file.write(ymin)
            csv_file.write(',')
            csv_file.write(xmax)
            csv_file.write(',')
            csv_file.write(ymax)
            csv_file.write('\n')
            

import os
import csv
from shapely.geometry import Polygon
from PIL import Image


# Each elements of this list is a list of inputs for local_IoU
local_IoU_inputs = []
# Open the CSV file for reading
with open('/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/predicted_results_V2.csv') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Iterate through each row in the CSV file
    for row in csv_reader:
        # We ignore the images without predictions
        if len(row) != 0:
            image_id = row[0]
            # We don't want the first line which is the legend
            if image_id == 'image_id':
                continue

            # Add the predicted box's coordinates to the new input for IoU
            x_min, y_min, x_max, y_max = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            # We want relative coordinates, not absolute coordinates (https://docs.ultralytics.com/guides/yolo-common-issues/#issues-related-to-model-predictions)
            # We need the dimensions of the image to normalize
            image_file_name = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/data/images/test/' + image_id + '.jpg'
            Image.open(image_file_name) # Open the image file
            width_total_image, height_total_image = image.size # Get the dimensions (size) of the image
            x_min, x_max = x_min / width_total_image, x_max / width_total_image # Divide x-coordinates by image width
            y_min, y_max = y_min / height_total_image, y_max / height_total_image # Divide y-coordinates by image height
            new_input = [x_min, y_min, x_max, y_max]

            # Detect to which true label it corresponds to
            folder_test_labels = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/data/labels/test'
            test_labels = [label for label in os.listdir(folder_test_labels)]
            # We use a counter to identify the true label it corresponds to
            image_id = int(row[0])
            counter = 1
            for label in test_labels: # Copy in the training dataset folder
                if image_id == counter:
                    with open(label) as txt_file:
                        line = txt_file.readline()[3:]
                        length = len(line)
                        xmin, ymin, xmax, ymax = '', '', '', ''
                        for index in range(length):
                            print(line[index])
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

                    new_input.append(float(xmin))
                    new_input.append(float(ymin))
                    new_input.append(float(xmax))
                    new_input.append(float(ymax))
                counter += 1
            
            # Add a new input to local_IoU (verify the size)
            if len(new_input) == 8:
                local_IoU_inputs.append(new_input)

print(local_IoU_inputs)





def local_IoU(
    xmin_pred_i,
    xmax_pred_i,
    ymin_pred_i,
    ymax_pred_i,
    xmin_true_i,
    xmax_true_i,
    ymin_true_i,
    ymax_true_i,
):
    """This function calculates the IoU for the image i.

    Args:
        xmin_pred_i: Value of the prediction min x-axis.
        xmax_pred_i: Value of the prediction max x-axis.
        ymin_pred_i: Value of the prediction min y-axis.
        ymax_pred_i: Value of the prediction max y-axis.
        xmin_true_i: Value of the true min x-axis.
        xmax_true_i: Value of the true max x-axis.
        ymin_true_i: Value of the true min y-axis.
        ymax_true_i: Value of the true max y-axis.

    Returns:
        The return value is the intersection over union.

    """
    if (xmin_true_i, xmax_true_i, ymin_true_i, ymax_true_i) == (0, 0, 0, 0):
        if (xmin_pred_i, xmax_pred_i, ymin_pred_i, ymax_pred_i) == (
            0,
            0,
            0,
            0,
        ):
            return 1

        else:
            return 0

    else:
        box_pred_i = [
            [xmin_pred_i, ymin_pred_i],
            [xmax_pred_i, ymin_pred_i],
            [xmax_pred_i, ymax_pred_i],
            [xmin_pred_i, ymax_pred_i],
        ]
        box_true_i = [
            [xmin_true_i, ymin_true_i],
            [xmax_true_i, ymin_true_i],
            [xmax_true_i, ymax_true_i],
            [xmin_true_i, ymax_true_i],
        ]
        poly_1 = Polygon(box_pred_i)
        poly_2 = Polygon(box_true_i)
        try:
            iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
            return iou
        except:
            return 0
        
local_IoU_results = [local_IoU(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]) for input in local_IoU_inputs]

for input in local_IoU_inputs:
    print(input)

print(f'{len(local_IoU_results)=}')
print(f'{local_IoU_results=}')
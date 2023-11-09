from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained models

# Use the model
results = model.train(data="/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/local_env/config.yaml", epochs=20, device='mps')  # train the model
# device='mps' for M1/M2 Apple chips
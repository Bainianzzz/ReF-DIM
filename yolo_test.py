from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11l.pt")  # load an official model

    # Validate the model
    metrics = model.val(data='./data.yaml')  # no arguments needed, dataset and settings remembered
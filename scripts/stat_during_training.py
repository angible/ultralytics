from ultralytics import YOLO

# Load the trained model
model = YOLO('Nrf_Central/yolo11s.yaml/20250124_095259/weights15-59/best.pt')  # Replace with your model file

# Validate on training data
results = model.val(data='data_config/0112.yaml', split='val', imgsz=704)
'''  
                            
                                        Use 'train or val' split
'''
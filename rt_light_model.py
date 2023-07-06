compressor = ModelCompressor(email='projecthourglass@protonmail.com', password = 'dummy') 

UPLOAD_MODEL_NAME = 'voc_v5'
TASK = Task.OBJECT_DETECTION
FRAMEWORK = Framework.PYTORCH
UPLOAD_MODEL_PATH = './yolov5_large.pt'
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [640,640]}]

model = compressor.upload_model(
        model_name=UPLOAD_MODEL_NAME,
        task=TASK,
        framework=FRAMEWORK,
        file_path=UPLOAD_MODEL_PATH,
        input_shapes=INPUT_SHAPES,
)


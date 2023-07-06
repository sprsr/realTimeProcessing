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

COMPRESSED_MODEL_NAME = '12norm_v5'
COMPRESSION_METHOD = CompressionMethod.PR_L2
RECOMMENDATION_METHOD = RecommendationMethod.SLAMP

for RATIO in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

    compressed_model = compressor.recommendation_compression(
            model_id=model.model_id,
            model_name=COMPRESSED_MODEL_NAME,
            compression_method=COMPRESSION_METHOD,
            recommendation_method=RECOMMENDATION_METHOD,
            recommendation_ratio=RATIO,
            output_path=f"./compressed_model/{RATIO}_compressed_pothole.pt",


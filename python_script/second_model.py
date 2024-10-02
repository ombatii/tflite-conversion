import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
import flatbuffers

# Load the Keras model
model_path = "../Keras_models/second_model.h5"
model = tf.keras.models.load_model(model_path)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = "../tflite models/second_model.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# Creates model info for the emotion classification model.
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "Emotion Classification Model"
model_meta.description = ("Classifies images into one of seven emotions: "
                          "angry, disgust, fear, happy, neutral, sad, surprise.")
model_meta.version = "v1"
model_meta.author = "Ombati"
model_meta.license = ("Apache License. Version 2.0 "
                      "http://www.apache.org/licenses/LICENSE-2.0.")

# Creates input info for the model.
input_meta = _metadata_fb.TensorMetadataT()
input_meta.name = "image"
input_meta.description = (
    "Input image to be classified. The expected image is {0} x {1}, with "
    "three channels (red, green, and blue) per pixel. Each value in the "
    "tensor is a single byte between 0 and 255.".format(150, 150))
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = (
    _metadata_fb.ColorSpaceType.RGB)
input_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.ImageProperties)

# Normalization process for input images
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (
    _metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [0.5]
input_normalization.options.std = [0.5]
input_meta.processUnits = [input_normalization]

# Statistics for the input
input_stats = _metadata_fb.StatsT()
input_stats.max = [255]
input_stats.min = [0]
input_meta.stats = input_stats

# Creates output info for the model.
output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "probability"
output_meta.description = "Probabilities of the seven emotion classes."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)

# Statistics for the output
output_stats = _metadata_fb.StatsT()
output_stats.max = [1.0]
output_stats.min = [0.0]
output_meta.stats = output_stats

# Associated labels for the output
label_file = _metadata_fb.AssociatedFileT()
label_file.name = os.path.basename("/Keras_models/labels.txt")
label_file.description = "Labels for emotions that the model can recognize."
label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
output_meta.associatedFiles = [label_file]

# Creates subgraph info.
subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]
model_meta.subgraphMetadata = [subgraph]

# Build the metadata buffer
b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()

# Populating the metadata
populator = _metadata.MetadataPopulator.with_model_file(tflite_model_path)
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files(["../Keras_models/labels.txt"])
populator.populate()

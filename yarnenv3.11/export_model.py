import coremltools as ct
from tensorflow.keras.models import load_model

keras_model = load_model('yarn_segmentation_model.h5')

# Define input types and preprocessing steps
input_image_type = ct.ImageType(scale=1/255.0,  # Example scaling factor for normalization
                                color_layout='RGB',  # Specifying the color layout if necessary
                                shape=(1, 256, 256, 3))  # Define the expected input shape

# Convert the model
mlmodel = ct.convert(keras_model, source="tensorflow", inputs=[input_image_type])

# Save the model
mlmodel.save('yarn_segmentation_model.mlpackage')


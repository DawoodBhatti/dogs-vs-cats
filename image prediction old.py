# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from os import getcwd, path

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img


# load an image and predict the class
def run_classification(image_path = ""):
    cwd = getcwd()
    
    image_path = path.join(cwd, 'tiny_dogs_vs_cats_vs_neither', 'cats', 'cat.0.jpg')
    model_path = path.join(cwd, 'model_checkpoints', 'model_epoch_17.keras')

    print(f"Loading image from: {image_path}")
    print(f"Loading model from: {model_path}")

    img = load_image(image_path)
    model = load_model(model_path)
    # predict the class
    result = model.predict(img)
    
    if result[0][0] > result[0][1] and result[0][0] > result[0][2]:
        class_type= "cat"
    elif result[0][1] > result[0][0] and result[0][1] > result[0][2]:
        class_type = "dog"
    else:
        class_type = "neither"
    
    print(class_type)
    
    return class_type
    
    
# entry point, run the example
run_classification()
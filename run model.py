#note: I switched from Python 3.13 to Python 3.12 to use keras
#tools -> interpreters
#C:/Users/Ali B/AppData/Local/Programs/Python/Python313/python.exe - previous main python
#C:/Users/Ali B/Documents/Python projects/dogs-vs-cats/virtual_env/Scripts/python.exe


# vgg16 model used for transfer learning on the dogs and cats dataset
import sys
from matplotlib import pyplot
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint  
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# define cnn model
def define_model():
   # load the pre-trained VGG16 model without the top layer
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
  
    # freeze all VGG16 layers
    for layer in model.layers:
        layer.trainable = False
        
    # add new classification layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    drop1 = Dropout(0.5)(class1)  # ✨ added dropout after dense layer
    output = Dense(3, activation='softmax')(drop1)
    
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()


def run_test_harness():
    # define model
    model = define_model()
    
    # create data generator
    datagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.mean = [123.68, 116.779, 103.939]

    # prepare iterators
    train_it = datagen.flow_from_directory('dataset_dogs_vs_cats_vs_neither/train/',
                                           class_mode='categorical', batch_size=64, target_size=(224, 224),
                                           shuffle=True)
    test_it = datagen.flow_from_directory('dataset_dogs_vs_cats_vs_neither/test/',
                                          class_mode='categorical', batch_size=64, target_size=(224, 224),
                                          shuffle=True)

    # ✅ define model checkpoint callback
    checkpoint = ModelCheckpoint(
        filepath='model_epoch_{epoch:02d}_valacc_{val_accuracy:.4f}.keras',
        save_freq='epoch',
        save_weights_only=False,
        save_best_only=False,
        verbose=1
    )

    # fit model with checkpoint callback
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it),
                        epochs=15, verbose=1, callbacks=[checkpoint])  # ✅ added callbacks

    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
    print('> %.3f' % (acc * 100.0))

    # learning curves
    summarize_diagnostics(history)

# entry point
run_test_harness()



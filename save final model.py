# save the final model to file
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os


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
    drop1 = Dropout(0.3)(class1)  # ✨ added dropout after dense layer
    output = Dense(3, activation='softmax')(drop1)
    
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# run the test harness for evaluating a model
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

    # prepare iterator
    train_it = datagen.flow_from_directory(
        'finalize_dogs_vs_cats_vs_neither/',
        class_mode='categorical',
        batch_size=64,
        target_size=(224, 224)
    )

    # create directory for checkpoints
    checkpoint_dir = 'model_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # define checkpoint callback
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras'),
        save_freq='epoch',
        save_weights_only=False,
        verbose=1
    )

    # fit model with checkpointing
    model.fit(
        train_it,
        steps_per_epoch=len(train_it),
        epochs=20,
        verbose=1,
        callbacks=[checkpoint]
    )

    # save final model
    model.save('final_model.keras')


# entry point, run the test harness
run_test_harness()
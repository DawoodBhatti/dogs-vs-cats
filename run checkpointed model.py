import sys
from matplotlib import pyplot
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint  
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# plot diagnostic learning curves
def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


def run_test_harness():
    # 🔁 Load model from checkpoint
    model = load_model('model_epoch_13_valacc_0.9694.keras')  
    
    # 🔧 Recompile model (in case optimizer state wasn't saved)
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # 📦 Data generator setup
    datagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.mean = [123.68, 116.779, 103.939]

    # 📁 Load training and test data
    train_it = datagen.flow_from_directory('dataset_dogs_vs_cats_vs_neither/train/',
                                           class_mode='categorical', batch_size=64, target_size=(224, 224),
                                           shuffle=True)
    test_it = datagen.flow_from_directory('dataset_dogs_vs_cats_vs_neither/test/',
                                          class_mode='categorical', batch_size=64, target_size=(224, 224),
                                          shuffle=True)

    # 💾 Checkpoint callback
    checkpoint = ModelCheckpoint(
        filepath='model_epoch_{epoch:02d}_valacc_{val_accuracy:.4f}.keras',
        save_freq='epoch',
        save_weights_only=False,
        save_best_only=False,
        verbose=1
    )


   # 🏋️ Resume training from epoch 13 to 20 (7 more epochs)
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it),
                        epochs=20, initial_epoch=13,
                        verbose=1, callbacks=[checkpoint])

    # 📊 Evaluate and plot
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
    print('> %.3f' % (acc * 100.0))
    summarize_diagnostics(history)
    

# 🚀 Entry point
run_test_harness()
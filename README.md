_Note: if you are just interested in running this program: go to release/single image prediction GUI.py or release/multiple image prediction.py_


# dogs-vs-cats

Build notes: I adapted this tutorial: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/ to create a 3 class classification program trained on a collection of ~ 36,000 images from https://www.kaggle.com/c/dogs-vs-cats/data and from Open Images V4. The model architechture follows VGG 16 and a transfer learning approach was used.

There are two ways to run this program. The first is a manual 'drag and drop' interface and the second lets you run classification on an entire folder to then provide output as txt/csv files in the selected folder.

three main branches in this repo:

1) build aka construction (messy mixing bowl)

2) dist is the finished product (finished cake)

3) release is the product in a gift-wrapped box (cake in a box handed to someone with a card)

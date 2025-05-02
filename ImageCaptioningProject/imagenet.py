from tensorflow.keras.applications.vgg19 import VGG19

model=VGG19(weights="imagenet")

model.save('vgg19.h5')


import tensorflow as tf

model = tf.keras.models.load_model("model/deepfake_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model/deepfake_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved: model/deepfake_model.tflite")

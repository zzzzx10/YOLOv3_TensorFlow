# Adaption to [wizyoung](https://github.com/wizyoung)/**[YOLOv3_TensorFlow](https://github.com/wizyoung/YOLOv3_TensorFlow)**  - Jul, 3,  2019

### 1. Introduction

As the predicted results of wizyoung/YOLOv3_Tensorflow model is the best of the models I've tried, I adapt it to accomplish fast loop testing of images on CPU mode.  This code can be further adapted to run in spark streaming.

- freeze the coco_dataset checkpoint to .pb file
- test images by .pb file (CPU mode)
- reduce the time of testing images
- spark streaming implementation: see [msp18034/Calories](https://github.com/msp18034/Calories)

### 2. Weights conversion

The pretrained darknet weights file can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights). Place this weights file under directory `./data/darknet_weights/` and then run:

```shell
python convert_weight.py
python freeze_graph.py
```

Then the converted TensorFlow checkpoint file and graph file will be saved to `./data/darknet_weights/` directory.

### 3. Test

Image test demo:

You can change image path in this .py file.

```shell
python test_image_2.py
```

### 4. References

[YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)

[qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)

https://github.com/wizyoung/YOLOv3_TensorFlow/issues/8


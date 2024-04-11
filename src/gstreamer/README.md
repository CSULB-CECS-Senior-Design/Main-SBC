# GStreamer based Object Tracking Example

This folder contains example code using [GStreamer](https://github.com/GStreamer/gstreamer) to
obtain camera images and perform image classification and object detection on the Edge TPU.

This code works on Linux using a webcam, Raspberry Pi with the Pi Camera, and on the Coral Dev
Board using the Coral Camera or a webcam. For the first two, you also need a Coral
USB/PCIe/M.2 Accelerator.


## Set up your device

First, be sure you have completed the [setup instructions for your Coral
device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
you have the latest software.
Importantly, you should have the latest TensorFlow Lite runtime installed
(as per the [Python quickstart](
https://www.tensorflow.org/lite/guide/python)). You can check which version is installed
using the ```pip3 show tflite_runtime``` command.


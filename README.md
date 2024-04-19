# R2-ARC Project

This repository is dedicated to a project that aims to create a droid similar to R2-D2. The droid is controlled via Bluetooth Low Energy from an iOS device and features autonomous following capabilities. These capabilities are powered by the [TensorFlow Lite API](https://tensorflow.org/lite) in conjunction with a Google Coral device, such as the [M.2 A+E key Accelerator](https://coral.ai/products/m2-accelerator-ae/) and provides an Object tracker for use with the detected objects.


## Installation

1.  First, be sure you have completed the [setup instructions for your Raspberry Pi 5 with Google Coral Edge TPU](https://gist.github.com/Reddimus/c6948d08a4f4b54ee9d075270bd79c3b). If it's been a while, repeat to be sure you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://gist.github.com/Reddimus/c6948d08a4f4b54ee9d075270bd79c3b)).

2.  Clone this Git repo onto your computer:

    ```
    mkdir main-sbc && cd main-sbc

    git clone https://github.com/CSULB-CECS-Senior-Design/Main-SBC.git
    ```

3.  Download the models:

    ```
    sh download_models.sh
    ```

    These models will be downloaded to a new folder
    ```models```.

4. Install the required Python packages:

    ```
    bash install_requirements.sh
    ```

## Test Object Detection

Now that you have the models and requirements installed:

1. Connect your video source (e.g. a webcam) to your Raspberry Pi 5.

2. Run the object detection script:

    ```bash
    cd src
    python3 vision.py
    ```

    This script will open a window showing the video feed with detected objects
    outlined.

> **Note:** Now that you have setup your Raspberry Pi 5 with Google Coral Edge TPU, you can use the [official Google Coral repository](https://github.com/google-coral/example-object-tracker).
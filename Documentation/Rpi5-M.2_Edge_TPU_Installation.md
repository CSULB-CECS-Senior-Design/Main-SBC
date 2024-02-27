# Raspberry Pi 5 - Google Coral Edge M.2 TPU installation guide

To get started with either the Mini PCIe or M.2 Accelerator, all you need to do is connect the card to your system, and then install our PCIe driver, Edge TPU runtime, and the TensorFlow Lite runtime. This page walks you through the setup and shows you how to run an example model.

The setup and operation is the same for both M.2 form-factors, including the M.2 Accelerator with Dual Edge TPU.

## Requirements

- Raspberry Pi 5 with the following Linux operating system: 
    - `Raspberry Pi OS (64-bit)` based on Debian 10 or newer
    - `Ubuntu (64-bit)` 23.10 or newer.
- All systems require support for MSI-X as defined in the PCI 3.0 specification
- At least one available M.2 module slot
- Python 3.6-3.9.16

## 1. Connect the module

1. Make sure the host system where you'll connect the module is shut down.
2. Carefully connect the M.2 module to the corresponding module slot on the host, according to your host system recommendations. We recommend the [PineBerry AI Hat (E-Key) for the Raspberry Pi 5](https://pineberrypi.com/products/hat-ai-for-raspberry-pi-5).
3. Updating your Pi:
    The boards we currently ship to customers support HAT+ power management for PCIe devices. If the power LED on your Pineberry Pi board does not light up when you connect it for the first time you need to update your firmware. Execute following commands in your Raspberry Pi OS terminal:
    ```bash
    sudo apt-get update && sudo apt-get upgrade
    ```
4. Enabling the PCIe interface:  
    Edit the available boot config file 
    ```bash
    sudo nano /boot/firmware/config.txt
    ```
    Within the config file add the following lines to the bottom of the file:
    ```bash
    # Enable the PCIe external connector
    dtparam=pciex1
    # Force Gen 3.0 speeds
    dtparam=pciex1_gen=3
    ```
    Save it, and reboot  

## 2: Install the PCIe driver and Edge TPU runtime

Next, you need to install both the Coral PCIe driver and the Edge TPU runtime. You can install these packages on your host computer as follows, on `Linux`.

The Coral ("Apex") PCIe driver is required to communicate with any Edge TPU device over a PCIe connection, whereas the Edge TPU runtime provides the required programming interface for the Edge TPU.

---

Before you install the PCIe driver on Linux, you first need to check whether you have a pre-built version of the driver installed. (Older versions of the driver have a bug that prevents updates and will result in failure when calling upon the Edge TPU.) So first follow these steps:

1. Check your Linux kernel version with this command:
    ```bash
    uname -r
    ```
    If it prints 4.18 or lower, you should be okay and can skip to begin installing our PCIe driver.
2. If your kernel version is 4.19 or higher, now check if you have a pre-build Apex driver installed:
    ```bash
    lsmod | grep apex
    ```
    If it prints nothing, then you're okay and continue to install our PCIe driver.  
    If it does print an Apex module name, stop here and follow the workaround to disable Apex and Gasket.

Now install the PCIe driver and runtime as follows:

1. First, add our Debian package repository to your system (be sure you have an internet connection):
    ```bash
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

    sudo apt-get update
    ```
2. Then install the PCIe driver and Edge TPU runtime packages:
    ```bash
    sudo apt-get install gasket-dkms libedgetpu1-std
    ```
3. If the user account you'll be using does not have root permissions, you might need to also add the following udev rule, and then verify that the "apex" group exists and that your user is added to it:
    ```bash
    sudo sh -c "echo 'SUBSYSTEM==\"apex\", MODE=\"0660\", GROUP=\"apex\"' >> /etc/udev/rules.d/65-apex.rules"

    sudo groupadd apex

    sudo adduser $USER apex
    ```
4. Now reboot the system.
5. Once rebooted, verify that the accelerator module is detected:  
    ```bash
    lspci -nn | grep 089a
    ```
    You should see something like this:  
    ```bash
    03:00.0 System peripheral: Device 1ac1:089a
    ```
    The `03` number and `System peripheral` name might be different, because those are host-system specific, but as long as you see a device listed with `089a` then you're okay to proceed.
6. Also verify that the PCIe driver is loaded:
    ```bash
    ls /dev/apex_0
    ```
    You should simply see the name repeated back:
    ```bash
    /dev/apex_0
    ```
    If the accelerator module is detected but `/dev/apex_0` is not found, then [read the troubleshooting section at the end of this guide](#pcie-devapex_0-driver-not-loaded).

7. Give permissions to the `/dev/apex_0` device by creating a new `udev` rule:  
    Open a terminal and use your favorite text editor with sudo to create a new file in /etc/udev/rules.d/. The file name should end with .rules. It's common practice to start custom rules with a higher number (e.g., 99-) to ensure they are applied after the default rules.
    For example:
    ```bash
    sudo nano /etc/udev/rules.d/99-coral-edgetpu.rules
    ```
8. Add a rule to the file:  
    You'll need to identify your device by attributes like `idVendor` and `idProduct` or use the `KERNEL` attribute if the device path is consistent. For the Coral Edge TPU, using the device path `/dev/apex_0` directly in a `udev` rule is not standard because this path might not be persistent across reboots or other device changes. Instead, use attributes to match the device.

    However, since we're dealing with a specific device path here, your rule might look something like this, assuming `/dev/apex_0` is consistently named and you're setting permissions:
    ```bash
    KERNEL=="apex_0", MODE="0666"
    ```
    This rule sets the device file `/dev/apex_0` to be readable and writable by everyone. Adjust the `MODE` as necessary for your security requirements.
9. Reload the `udev` rules and trigger them:
    After saving the file, you need to reload the `udev` rules and trigger them to apply the changes without rebooting.
    Reload the rules:
    ```bash
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ```
10. Verify:  
    Reboot your system to verify that the permissions for `/dev/apex_0` are set as expected. After rebooting, check the permissions of the device file:
    ```bash
    ls -l /dev/apex_0
    ```

    Note: The use of MODE="0666" makes the device world-readable and writable, which may not be secure for all environments. Consider your security requirements and adjust the permissions accordingly, possibly using GROUP to restrict access to users within a specific group.

Now continue to install PyCoral and TensorFlow Lite.

## 3: Install the PyCoral library

PyCoral is a Python library built on top of the TensorFlow Lite library to speed up your development and provide extra functionality for the Edge TPU.

We recommend you start with the PyCoral API, and we use this API in our example code below, because it simplifies the amount of code you must write to run an inference. But you can build your own projects using TensorFlow Lite directly, in either Python or C++.

First check your Linux system's Python version:

```bash
python3 --version
```
PyCoral currently supports Python 3.6 through 3.9.16. If your default version is something else, we suggest you [install Python 3.9 with pyenv](https://realpython.com/intro-to-pyenv/).

To install the PyCoral library, use the following commands based on your Python environment.

### On Linux with System Python 3.6-3.9.16

```bash
sudo apt-get install python3-pycoral
```

### On Linux with Python 3.6-3.9.16 installed with pyenv

```bash
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

Lastly, check the list of installed packages to verify that PyCoral is installed:

```bash
pip3 list
```

The packages needed should roughly look like this:

```bash
Python 3.9.16

Package        Version
----------------------------
numpy            1.26.4
Pillow           9.5.0
pip              24.0
pycoral          2.0.0
setuptools       58.1.0
tflite-runtime   2.5.0.post1
```

    Note: Pillow must be 9.5.0 or older. If you have a newer version, you can downgrade it with `pip3 install Pillow==9.5.0`.

## 3: Run a model on the Edge TPU

Now you're ready to run an inference on the Edge TPU.

Follow these steps to perform image classification with our example code and MobileNet v2:

1. Download the example code from GitHub:
    ```bash
    mkdir coral && cd coral

    git clone https://github.com/google-coral/pycoral.git

    cd pycoral
    ```
2. Download the model, labels, and bird photo:
    ```bash
    bash examples/install_requirements.sh classify_image.py
    ```
3. Run the image classifier with the bird photo (shown in figure 1):
    ```bash
    python3 examples/classify_image.py \
    --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
    --labels test_data/inat_bird_labels.txt \
    --input test_data/parrot.jpg
    ```

You should see results like this:
```bash
----INFERENCE TIME----
Note: The first inference on Edge TPU is slow because it includes loading the model into Edge TPU memory.
11.8ms
2.9ms
2.8ms
2.9ms
2.9ms
-------RESULTS--------
Ara macao (Scarlet Macaw): 0.75781
```

These speeds are faster compared to running the same model on the Google Coral USB TPU:
```bash
----INFERENCE TIME----
Note: The first inference on Edge TPU is slow because it includes loading the model into Edge TPU memory.
20.6ms
7.0ms
6.8ms
5.2ms
5.1ms
-------RESULTS--------
Ara macao (Scarlet Macaw): 0.75781
```

Congrats! You just performed an inference on the Edge TPU using TensorFlow Lite.

To demonstrate varying inference speeds, the example repeats the same inference five times. Your inference speeds might differ based on your host system.

The top classification label is printed with the confidence score, from 0 to 1.0.

To learn more about how the code works, take a look at the `classify_image.py` [source code](https://github.com/google-coral/pycoral/blob/master/examples/classify_image.py) and read about how to run [inference with TensorFlow Lite](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python).

Note: The example above uses the PyCoral API, which calls into the TensorFlow Lite Python API, but you can instead directly call the TensorFlow Lite Python API or use the TensorFlow Lite C++ API. For more information about these options, read the Edge [TPU inferencing overview](https://coral.ai/docs/edgetpu/inference/).

## Next steps

Important: To sustain maximum performance, the Edge TPU must remain below the maximum operating temperature specified in the datasheet. By default, if the Edge TPU gets too hot, the PCIe driver slowly reduces the operating frequency and it may reset the Edge TPU to avoid permanent damage. To learn more, including how to configure the frequency scaling thresholds, read how to [manage the PCIe module temperature](https://coral.ai/docs/pcie-parameters/).

To run some other models, such as real-time object detection, pose estimation, keyphrase detection, on-device transfer learning, and others, check out our [example projects](https://coral.ai/examples/). In particular, if you want to try running a model with camera input, try one of the several [camera examples](https://github.com/google-coral/examples-camera).

If you want to train your own model, try these tutorials:
- [Retrain an image classification model using post-training quantization](https://colab.sandbox.google.com/github/google-coral/tutorials/blob/master/retrain_classification_ptq_tf1.ipynb) (runs in Google Colab)
- [Retrain an image classification model using quantization-aware training](https://coral.ai/docs/edgetpu/retrain-classification/) (runs in Docker)
- [Retrain an object detection model using quantization-aware training](https://coral.ai/docs/edgetpu/retrain-detection/) (runs in Docker)
Or to create your own model that's compatible with the Edge TPU, read [TensorFlow Models on the Edge TPU](https://coral.ai/docs/edgetpu/models-intro/).

The following section describes how the power throttling works and how to customize the trip points.

## Troubleshooting

Here are some solutions to possible problems on Linux.

### HIB error
If you receive an error messages such as the following when you run an inference...
```bash
HIB Error. hib_error_status = 0000000000002200, hib_first_error_status = 0000000000000200
```
... You should be able to solve it if you modify your kernel command line arguments to include `gasket.dma_bit_mask=32`.

For information about how to modify your kernel command line arguments, refer to your respective platform documentation. For bootloaders based on U-Boot, you can usually modify the arguments either by modifying the `bootargs` U-Boot environment variable or by setting othbootargs environment variable as follows:
```bash
=> setenv othbootargs gasket.dma_bit_mask=32
=> printenv othbootargs
othbootargs=gasket.dma_bit_mask=32
=> saveenv
```
If you make the above change and then receive errors such as, `DMA: Out of SW-IOMMU space`, then you need to increase the `swiotlb` buffer size by adding another kernel command line argument: `swiotlb=65536`.

### pcieport error

If you see a lot of errors such as the following:
```bash
pcieport 0000:00:01.0: PCIe Bus Error: severity=Corrected, type=Data Link Layer, id=0008(Transmitter ID)
pcieport 0000:00:01.0: device [10de:0fae] error status/mask=00003100/00002000
pcieport 0000:00:01.0: [ 8] RELAY_NUM Rollover
pcieport 0000:00:01.0: [12] Replay Timer Timeout
pcieport 0000:00:01.0: PCIe Bus Error: severity=Uncorrected (Non-Fatal), type=Transaction Layer, id=0008(Requester ID)
pcieport 0000:00:01.0: device [10de:0fae] error status/mask=00004000/00000000
```
... You should be able to solve it if you modify your kernel command line arguments to include `pcie_aspm=off`.

For information about how to modify your kernel command line arguments, refer to your respective platform documentation. If your device includes U-Boot, see the previous [HIB error](https://coral.ai/docs/m2/get-started/#hib-error) for an example of how to modify the kernel commands. For certain other devices, you might instead add `pcie_aspm=off` to an `APPEND` line in your system `/boot/extlinux/extlinux.conf` file:
```bash
LABEL primary
      MENU LABEL primary kernel
      LINUX /boot/Image
      INITRD /boot/initrd
      APPEND ${cbootargs} quiet pcie_aspm=off
```

### Workaround to disable Apex and Gasket

The following procedure is necessary only if your system includes a pre-build driver for Apex devices (as per the first steps for [installing the PCIe driver](https://coral.ai/docs/m2/get-started/#1-install-the-pcie-driver)). Due to a bug, updating this driver with ours can fail, so you need to first disable the `apex` and `gasket` modules as follows:

1. Create a new file at /etc/modprobe.d/blacklist-apex.conf and add these two lines:
    ```bash
    blacklist gasket
    blacklist apex
    ```
2. Reboot the system.
3. Verify that the apex and gasket modules did not load by running this:
    ```bash
    lsmod | grep apex
    ```
    It should print nothing.
4. Now follow the rest of the steps to [install the PCIe driver](https://coral.ai/docs/m2/get-started/#1-install-the-pcie-driver).
5. Finally, delete `/etc/modprobe.d/blacklist-apex.conf` and reboot your system.

### PCIe (dev/apex_0) driver not loaded

Ensure the Kernel Moudle is Loaded

Check if the `gasket` and `apex` kernel modules are loaded properly. You've already checked for `apex`, but let's ensure everything is set up correctly:

```bash
lsmod | grep gasket
lsmod | grep apex
```

If they're not listed, try manually loading them:

```bash
sudo modprobe gasket
sudo modprobe apex
```

If you see the following error message continue reading:

```bash
modprobe: FATAL: Module gasket not found in directory /lib/modules/6.1.0-rpi8-rpi-2712
modprobe: FATAL: Module apex not found in directory /lib/modules/6.1.0-rpi8-rpi-2712
```

The error messages from `modprobe` indicate that the `gasket` and `apex` modules are not found in your current kernel's module directory. This suggests that either the modules are not installed correctly, or they are not compatible with your current kernel version (6.1.0 for Raspberry Pi). Here are some steps you can take to address this issue:

1. Ensure Kernel Headers are Installed
    For modules like `gasket` and `apex` to be built and installed properly, you need the kernel headers for your currently running kernel. Install the kernel headers with:
    ```bash
    sudo apt-get install raspberrypi-kernel-headers
    ```
    After installing the headers, try reinstalling the gasket-dkms and libedgetpu1-std packages, as DKMS should automatically build the modules against your current kernel:
    ```bash
    sudo apt-get reinstall gasket-dkms libedgetpu1-std
    ```
2. Check DKMS Status
    After installing the kernel headers and reinstalling the packages, check the status of DKMS to see if the `gasket` and `apex` modules have been built:
    ```bash
    dkms status
    ```
    This command will list all DKMS modules and their status. You're looking for gasket and apex to be listed as installed for your kernel version.

All done you should roughly see the following output:
```bash
Deprecated feature: REMAKE_INITRD (/var/lib/dkms/gasket/1.0/source/dkms.conf)
Deprecated feature: REMAKE_INITRD (/var/lib/dkms/gasket/1.0/source/dkms.conf)
gasket/1.0, 6.1.0-rpi8-rpi-2712, aarch64: installed
gasket/1.0, 6.1.0-rpi8-rpi-v8, aarch64: installed
```

You can now continue with the rest of the steps to [install the PCIe driver](#2-install-the-pcie-driver-and-edge-tpu-runtime).
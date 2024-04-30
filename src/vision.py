"""
vision.py
R2-ARC Machine Vision module using gstreamer, Google Coral TPU, and TensorFlow Lite for object detection.
"""

import time, numpy, os
import motors
from gstreamer import *
from gstreamer import Object as Object

class InfraRedSensors:
    def __init__(self) -> None:
        self.MAX_DISTANCE = numpy.uint8(80)
        self.MIN_DISTANCE = numpy.uint8(5)
        self.TOO_CLOSE_DISTANCE = numpy.uint8(40)
        self.front_distance = self.MAX_DISTANCE
        self.back_distance = self.MAX_DISTANCE
    
    def get_front_distance(self) -> numpy.uint8:
        return self.front_distance
    
    def get_back_distance(self) -> numpy.uint8:
        return self.back_distance

    def is_front_too_close(self) -> bool:
        return self.front_distance < self.TOO_CLOSE_DISTANCE
    
    def is_back_too_close(self) -> bool:
        return self.back_distance < self.TOO_CLOSE_DISTANCE
    
    def decode_recieved_data(self, data) -> str:
        """
        Splits an unsigned 8 bit integer into two unsigned "4 bit" integers.
        """
        msbs = numpy.uint8(data >> 4)
        lsbs = numpy.uint8(data & 0x0F)
        return (msbs, lsbs)
    
    def update_distances(self, data) -> None:
        """
        Updates the front and back distance values.
        """
        self.front_distance, self.back_distance = self.decode_recieved_data(data)

class PositionSide:
    LEFT = False
    RIGHT = True

class AutoMovements:
    def __init__(self, motor: motors.Movements):
        self.last_human_position = PositionSide.LEFT    # Default
        self._motors = motor
        self._ir_sensors = InfraRedSensors()

    def _get_obj_xside(self, obj: Object) -> PositionSide:
        """Returns the position side of the object in the camera view.
        Args:
            obj (BBox Object): The object detected in the camera view.
        Returns:
            PositionSide: The side of the object in the camera view; LEFT or RIGHT.
        """
        xcenter = (obj.bbox.xmin + obj.bbox.xmax) / 2
        return PositionSide.LEFT if xcenter < 0.5 else PositionSide.RIGHT

    def _face_last_human_position(self):
        """Pivots in the direction of the last seen human position."""
        if self.last_human_position == PositionSide.LEFT:
            # print("Last seen human position was left, pivoting left")
            recieved_data = recieved_data = self._motors.pivot_left()
        else:
            # print("Last seen human position was right, pivoting right")
            recieved_data = self._motors.pivot_right()
        
        self._ir_sensors.update_distances(recieved_data)

    def _follow_human(self, human) -> None:
        """Follows the given human in the camera view.
        Args:
            human (BBox Object): The human detected in the camera view.
        """
        # if human is centered in camera view
        if human.bbox.xmin < 0.5 and 0.5 < human.bbox.xmax:
            # print("Human is centered, moving forward")
            recieved_data = self._motors.forward()
        # Else if human is on left side of midpoint
        elif human.bbox.xmax < 0.5:
            # print("Human is to the left of the center, moving left")
            recieved_data = self._motors.left()
        # Else human is on right side of midpoint
        else:
            # print("Human is to the right of the center, moving right")
            recieved_data = self._motors.right()

        self._ir_sensors.update_distances(recieved_data)

    def find_human(self, objs: list[Object]) -> bool:
        """Finds the closest human and object in the camera view. Then moves towards the human.
        Args:
            objs (list[Object]): The Bounding Box objects detected in the camera view.
        Returns:
            bool: True if human is found/reached, False otherwise.
        """
        reached: bool = False
        closest_human = detect.get_closest_obj(objs=[obj for obj in objs if obj.id == 0])
        closest_obj = detect.get_closest_obj(objs=objs, min_certainty=None)

        # If no human is detected
        if not closest_human:
            # print("No humans detected")
            # Pivot in directions of last seen human position
            self._face_last_human_position()
            return reached

        # If human is the closest object
        if closest_human == closest_obj:
            # print("Closest object is a human")

            if detect.is_too_close(closest_human) or self._ir_sensors.is_front_too_close():
                # print("Human is too close to the camera")
                recieved_data = self._motors.stop()
                self._ir_sensors.update_distances(recieved_data)
                reached = True
            # Else human is not too close to the camera
            else:
                self._follow_human(closest_human)
        # Else closest object is not a human
        else:
            # print(f"Closest object is not a human. Object ID: {closest_obj.id}")
            # print("Avoiding foreign object")

            if detect.is_too_close(closest_obj) or self._ir_sensors.is_front_too_close():
                # print("Object is too close to the camera")
                if not self._ir_sensors.is_back_too_close():
                    # print("Backing up")
                    recieved_data = self._motors.backward()
                    self._ir_sensors.update_distances(recieved_data)
                self._face_last_human_position()
            else:  # steer into nearest human
                self._follow_human(closest_human)

        # print()
        # Cache last seen human position
        self.last_human_position = self._get_obj_xside(closest_human)
        return reached

class DroidVision:
    def __init__(self, 
                 motor: motors.Movements, 
                 model: str = "../models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", 
                 labels: str = "../models/coco_labels.txt", 
                 top_k = numpy.uint8(20), 
                 tracker = None, 
                 threshold = numpy.float16(0.2), 
                 videosrc: str = '/dev/video0', 
                 videofmt: str = 'raw', 
                 resolution: tuple = cameras.get_resolution()):
        """"Main function to run object detection on camera frames using GStreamer.
        Args:
            model (str, optional): The path to the model file. Defaults to "../models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite".
            labels (str, optional): The path to the labels file. Defaults to "../models/coco_labels.txt".
            top_k (int, optional): The number of top results to display. Defaults to 20.
            tracker ([type], optional): The object tracker to use. Defaults to None. Choices: [None, 'sort']
            threshold (float, optional): The threshold for detection. Defaults to 0.2.
            videosrc (str, optional): The video source. Defaults to '/dev/video0'.
            videofmt (str, optional): The video format. Defaults to 'raw'. Choices: ['raw', 'h264', 'jpeg']
            resolution (tuple, optional): The resolution of the camera. Defaults to cameras.get_resolution().
        """
        self.model = model
        self.labels = labels
        self.top_k = top_k
        self.tracker = tracker
        self.threshold = threshold
        self.videosrc = videosrc
        self.videofmt = videofmt
        self.resolution = resolution
        self._init_model()
        self._init_display()
        self.follow: bool = False
        self.follow_counter = numpy.uint8(0)
        self.automove = AutoMovements(motor)

    def _init_model(self):
        attempts = numpy.uint8(3)
        while attempts:
            try:
                print('Loading {} with {} labels.'.format(self.model, self.labels))
                self.interpreter = common.make_interpreter(self.model)
                self.interpreter.allocate_tensors()
                self.labels = detect.load_labels(self.labels)
                break
            except Exception as e:
                print(f"Error initializing model: {e}")
                attempts -= numpy.uint8(1)

    def _init_display(self):
        w, h, _ = common.input_image_size(self.interpreter)
        self.inference_size = (w, h)
        # Average fps over last 30 frames.
        self.fps_counter = common.avg_fps_counter(30)

    def _auto_stop(self, reached_human: bool) -> bool:
        """Implement a counter to automatically stop following after a certain number of frames of detecting a human.
        Args:
            reached_human (bool): True if human is found/reached, False otherwise.
        Returns:
            bool: True if the counter reaches the counter's data type limit, False otherwise.
        """
        if reached_human and self.follow_counter < numpy.iinfo(self.follow_counter.dtype).max:
            self.follow_counter += numpy.uint8(1)
        elif not reached_human and self.follow_counter > numpy.iinfo(self.follow_counter.dtype).min:
            self.follow_counter -= numpy.uint8(1)
        return self.follow_counter == numpy.iinfo(self.follow_counter.dtype).max

    def _user_callback(self, input_tensor, src_size, inference_box, mot_tracker):
        start_time = time.monotonic()
        common.set_input(self.interpreter, input_tensor)
        self.interpreter.invoke()
        objs = detect.get_output(self.interpreter, self.threshold, self.top_k)
        # print(f"Follow state: {self.follow}")
        if self.follow:
            reached_human = self.automove.find_human(objs)
            self.follow = not self._auto_stop(reached_human)
            # if not self.follow:
                # print("Auto stopped triggered")
        end_time = time.monotonic()
        # print(f"Detected objects: {objs}")
        detections = numpy.array([(o.bbox.xmin, o.bbox.ymin, o.bbox.xmax, o.bbox.ymax, o.score) for o in objs])
        trdata = []
        trackerFlag = False
        if detections.any():
            if mot_tracker != None:
                trdata = mot_tracker.update(detections)
                trackerFlag = True
            text_lines = [
                'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
                'FPS: {} fps'.format(round(next(self.fps_counter))), ]
        if len(objs) != 0:
            return detect.generate_svg(src_size, self.inference_size, inference_box, objs, self.labels, text_lines, trdata, trackerFlag)

    def start(self):
        self.run = gstreamer.run_pipeline(
            self._user_callback,
            self.resolution,
            self.inference_size,
            self.tracker,
            self.videosrc,
            self.videofmt
        )

    def stop(self, process: str = "vision.py"):
        self.follow = False
        self.run = None
        os.system(f"pkill -f {process}")

    def set_follow(self, follow: bool):
        self.follow = follow
        self.follow_counter = numpy.uint8(0)

    def toggle_follow(self):
        self.follow ^= True
        self.follow_counter = numpy.uint8(0)
        if not self.follow:
            self.automove._motors.stop()

if __name__ == '__main__':
    # Test the DroidVision class
    import spidev
    # Setup spi communication
    spi = spidev.SpiDev()
    spi.open(bus=0, device=0)
    spi.max_speed_hz = 5_000_000
    # Setup motor controller communication
    r2motor = motors.Movements(spi)
    # Setup DroidVision
    vision = DroidVision(tracker='sort', motor=r2motor)
    # vision = DroidVision(motor=r2motor)
    vision.toggle_follow()
    vision.start()

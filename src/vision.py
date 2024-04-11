"""
vision.py
"""

import time, numpy
import motors
from gstreamer import *
from gstreamer import Object as Object

class PositionSide:
    LEFT = False
    RIGHT = True

class AutoMovements:
    def __init__(self, motor: motors.Movements):
        self.last_human_position = PositionSide.LEFT    # Default
        self._motors = motor

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
            print("Last seen human position was left, pivoting left")
            self._motors.pivot_left()
        else:
            print("Last seen human position was right, pivoting right")
            self._motors.pivot_right()

    def _follow_human(self, human) -> None:
        """Follows the given human in the camera view.
        Args:
            human (BBox Object): The human detected in the camera view.
        """
        # if human is centered in camera view
        if human.bbox.xmin < 0.5 and 0.5 < human.bbox.xmax:
            print("Human is centered, moving forward")
            self._motors.forward()
        # Else if human is on left side of midpoint
        elif human.bbox.xmax < 0.5:
            print("Human is to the left of the center, moving left")
            self._motors.left()
        # Else human is on right side of midpoint
        elif human.bbox.xmin > 0.5:
            print("Human is to the right of the center, moving right")
            self._motors.right()

    def find_human(self, objs: list[Object]) -> None:
        """Finds the closest human and object in the camera view. Then moves towards the human.
        Args:
            objs (list[Object]): The Bounding Box objects detected in the camera view.
        """
        closest_human = detect.get_closest_obj(objs=[obj for obj in objs if obj.id == 0])
        closest_obj = detect.get_closest_obj(objs=objs, min_certainty=None)

        # If no human is detected
        if not closest_human:
            print("No humans detected")
            # Pivot in directions of last seen human position
            self._face_last_human_position()
            return

        # If human is the closest object
        if closest_human == closest_obj:
            print("Closest object is a human")

            if detect.is_too_close(closest_human):
                print("Human is too close to the camera")
                self._motors.stop()
            # Else human is not too close to the camera
            else:
                self._follow_human(closest_human)
        # Else closest object is not a human
        else:
            print(f"Closest object is not a human. Object ID: {closest_obj.id}")
            # print("Avoiding foreign object")

            if detect.is_too_close(closest_obj):
                print("Object is too close to the camera")
                self._face_last_human_position()
            else:  # steer into nearest human
                self._follow_human(closest_human)

        # Cache last seen human position
        self.last_human_position = self._get_obj_xside(closest_human)

class DroidVision:
    def __init__(self, 
                 motor: motors.Movements = motors.Movements(), 
                 model: str = "../models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", 
                 labels: str = "../models/coco_labels.txt", 
                 top_k: int = 20, 
                 tracker = None,
                 threshold: float = 0.2, 
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
            resolution (tuple, optional): The resolution of the camera. Defaults to cameras.get_resolution()."""
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
        self.automove = AutoMovements(motor)

    def _init_model(self):
        print('Loading {} with {} labels.'.format(self.model, self.labels))
        self.interpreter = common.make_interpreter(self.model)
        self.interpreter.allocate_tensors()
        self.labels = detect.load_labels(self.labels)

    def _init_display(self):
        w, h, _ = common.input_image_size(self.interpreter)
        self.inference_size = (w, h)
        # Average fps over last 30 frames.
        self.fps_counter = common.avg_fps_counter(30)

    def _user_callback(self, input_tensor, src_size, inference_box, mot_tracker):
        start_time = time.monotonic()
        common.set_input(self.interpreter, input_tensor)
        self.interpreter.invoke()
        objs = detect.get_output(self.interpreter, self.threshold, self.top_k)
        # print(f"Follow state: {self.follow}")
        if self.follow:
            self.automove.find_human(objs)
        end_time = time.monotonic()
        detections = [(o.bbox.xmin, o.bbox.ymin, o.bbox.xmax, o.bbox.ymax, o.score) for o in objs]
        # for obj in objs:
        #     print(f"Object: label={self.labels[obj.id]}, score={obj.score}, area={obj.bbox.area}, ymin={obj.bbox.ymin}, ymax={obj.bbox.ymax}")
        # print()
        detections = numpy.array(detections)
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
        
    def start(self, max_attempts: int = 3):
        attempts = 0
        while attempts < max_attempts:
            try:
                self.run = gstreamer.run_pipeline(self._user_callback, 
                             self.resolution, 
                             self.inference_size, 
                             self.tracker, 
                             self.videosrc, 
                             self.videofmt)
                attempts = max_attempts
            except Exception as e:
                attempts += 1
                print(f"Error: {e}, attempts: {attempts}")
        
    def set_follow(self, follow: bool):
        self.follow = follow

    def toggle_follow(self):
        self.follow ^= True

if __name__ == '__main__':
    vision = DroidVision(tracker='sort', resolution=cameras.get_razer_kiyo_resolution())
    # vision = DroidVision()
    vision.toggle_follow()
    vision.start()

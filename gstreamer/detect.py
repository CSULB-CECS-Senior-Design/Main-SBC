"""
A demo which runs object detection on camera frames using GStreamer.
It also provides support for Object Tracker.

Run default object detection:
python3 detect.py

Choose different camera and input encoding
python3 detect.py --videosrc /dev/video1 --videofmt jpeg

Choose an Object Tracker. Example : To run sort tracker
python3 detect.py --tracker sort

TEST_DATA=../all_models

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
"""
import argparse
import collections
import common
import gstreamer
import numpy as np
import os
import re
import svgwrite
import time
import cameras
from tracker import ObjectTracker


Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])


def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
        return {int(num): text.strip() for num, text in lines}


def shadow_text(dwg, x, y, text, font_size=20):
    dwg.add(dwg.text(text, insert=(x+1, y+1), fill='black', font_size=font_size))
    dwg.add(dwg.text(text, insert=(x, y), fill='white', font_size=font_size))


def generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines, trdata, trackerFlag):
    dwg = svgwrite.Drawing('', size=src_size)
    src_w, src_h = src_size
    inf_w, inf_h = inference_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h

    for y, line in enumerate(text_lines, start=1):
        shadow_text(dwg, 10, y*20, line)
    if trackerFlag and (np.array(trdata)).size:
        for td in trdata:
            x0, y0, x1, y1, trackID = td[0].item(), td[1].item(
            ), td[2].item(), td[3].item(), td[4].item()
            overlap = 0
            for ob in objs:
                dx0, dy0, dx1, dy1 = ob.bbox.xmin.item(), ob.bbox.ymin.item(
                ), ob.bbox.xmax.item(), ob.bbox.ymax.item()
                area = (min(dx1, x1)-max(dx0, x0))*(min(dy1, y1)-max(dy0, y0))
                if (area > overlap):
                    overlap = area
                    obj = ob

            # Relative coordinates.
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            # Absolute coordinates, input tensor space.
            x, y, w, h = int(x * inf_w), int(y *
                                             inf_h), int(w * inf_w), int(h * inf_h)
            # Subtract boxing offset.
            x, y = x - box_x, y - box_y
            # Scale to source coordinate space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {} ID:{}'.format(
                percent, labels.get(obj.id, obj.id), int(trackID))
            shadow_text(dwg, x, y - 5, label)
            dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                             fill='none', stroke='red', stroke_width='2'))
    else:
        for obj in objs:
            x0, y0, x1, y1, a = list(obj.bbox)
            # Relative coordinates.
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            # Absolute coordinates, input tensor space.
            x, y, w, h = int(x * inf_w), int(y *
                                             inf_h), int(w * inf_w), int(h * inf_h)
            # Subtract boxing offset.
            x, y = x - box_x, y - box_y
            # Scale to source coordinate space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
            shadow_text(dwg, x, y - 5, label)
            dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                             fill='none', stroke='red', stroke_width='2'))
    return dwg.tostring()


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax', 'area'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()


def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    category_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(category_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax),
                      area=(xmax - xmin) * (ymax - ymin)))
    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]


class PositionSide:
    LEFT = False
    RIGHT = True


class Movements:
    def __init__(self):
        self.last_human_position = PositionSide.LEFT    # Default

    def get_obj_xside(self, obj: Object) -> PositionSide:
        """Returns the position side of the object in the camera view.
        Args:
            obj (BBox Object): The object detected in the camera view.
        Returns:
            PositionSide: The side of the object in the camera view; LEFT or RIGHT.
        """
        xcenter = (obj.bbox.xmin + obj.bbox.xmax) / 2
        return PositionSide.LEFT if xcenter < 0.5 else PositionSide.RIGHT

    def face_last_human_position(self):
        """Pivots in the direction of the last seen human position."""
        if self.last_human_position == PositionSide.LEFT:
            print("Last seen human position was left")
            # pivot_left()
        else:
            print("Last seen human position was right")
            # pivot_right()

    def find_closest_objs(self, objs: list[Object], min_certainty: float = 0.5) -> tuple:
        """Finds the closest human and object in the camera view.
        Args:
            min_certainty (float, optional): The minimum certainty required for an object to be considered. Defaults to 0.5.
        Returns:
            tuple: (closest_human, closest_obj)
        """
        objs = [obj for obj in objs if obj.score > min_certainty]

        closest_human, closest_obj = None, None
        closest_human_area, closest_obj_area = 0, 0
        for obj in objs:
            # If object detected is outside of camera view increase area
            if obj.bbox.ymin < 0.1 and 0.9 < obj.bbox.ymax:
                obj.bbox.area *= 1.1
            # Update closest human
            if obj.id == 0 and obj.bbox.area > closest_human_area:
                closest_human = obj
                closest_human_area = obj.bbox.area
            # Update closest object
            if obj.bbox.area > closest_obj_area:
                closest_obj = obj
                closest_obj_area = obj.bbox.area

        return (closest_human, closest_obj)

    def is_too_close(self, obj, threshold=0.2) -> bool:
        """Checks if the given object is too close to the camera.
        Args:
            obj (BBox Object): The object detected in the camera view.
            threshold (float, optional): The threshold for closeness. Defaults to 0.2.
        Returns:
            bool: True if object is too close to the camera, False otherwise.
        """
        return obj.bbox.xmin < threshold and 1 - threshold < obj.bbox.xmax

    def follow_human(self, human) -> None:
        """Follows the given human in the camera view.
        Args:
            human (BBox Object): The human detected in the camera view.
        """
        # if human is centered in camera view
        if human.bbox.xmin < 0.5 and 0.5 < human.bbox.xmax:
            print("Human is centered")
            # move_forward()
        # Else if human is on left side of midpoint
        elif human.bbox.xmax < 0.5:
            print("Human is to the left of the center")
            # move_left()
        # Else human is on right side of midpoint
        elif human.bbox.xmin > 0.5:
            print("Human is to the right of the center")
            # move_right()

    def find_human(self, objs: list[Object]) -> None:
        """Finds the closest human and object in the camera view. Then moves towards the human.
        Args:
            objs (list[Object]): The Bounding Box objects detected in the camera view.
        """
        closest_human, closest_obj = self.find_closest_objs(objs)

        # If no human is detected
        if not closest_human:
            print("No humans detected")
            # Pivot in directions of last seen human position
            self.face_last_human_position()
            return

        # If human is the closest object
        if closest_human == closest_obj:
            print("Closest object is a human")
            # print(f"Centering on human with area {closest_human.bbox.area}")
            # If human is not too close to the camera
            if not self.is_too_close(closest_human):
                self.follow_human(closest_human)
            # Else goal met; stop moving
        # Else closest object is not a human
        else:
            print(f"Closest object is not a human. Object ID: {closest_obj.id}")
            # print("Avoiding foreign object")

            if self.is_too_close(closest_obj):
                print("Object is too close to the camera")
                self.face_last_human_position()
            else:  # steer into nearest human
                self.follow_human(closest_human)

        # Cache last seen human position
        self.last_human_position = self.get_obj_xside(closest_human)


def main():
    default_model_dir = '../models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video0')
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    parser.add_argument('--tracker', help='Name of the Object Tracker To be used.',
                        default=None,
                        choices=[None, 'sort'])
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    w, h, _ = common.input_image_size(interpreter)
    inference_size = (w, h)
    # Average fps over last 30 frames.
    fps_counter = common.avg_fps_counter(30)
    moves = Movements()

    def user_callback(input_tensor, src_size, inference_box, mot_tracker):
        nonlocal fps_counter
        start_time = time.monotonic()
        common.set_input(interpreter, input_tensor)
        interpreter.invoke()
        # For larger input image sizes, use the edgetpu.classification.engine for better performance
        objs = get_output(interpreter, args.threshold, args.top_k)
        moves.find_human(objs)
        end_time = time.monotonic()
        detections = []  # np.array([])
        for n in range(0, len(objs)):
            element = []  # np.array([])
            element.append(objs[n].bbox.xmin)
            element.append(objs[n].bbox.ymin)
            element.append(objs[n].bbox.xmax)
            element.append(objs[n].bbox.ymax)
            print(f"Object[{n}]: label={labels[objs[n].id]}, score={objs[n].score}, area={objs[n].bbox.area}")
            element.append(objs[n].score)  # print('element= ',element)
            detections.append(element)  # print('dets: ',dets)
        print()
        # convert to numpy array #      print('npdets: ',dets)
        detections = np.array(detections)
        trdata = []
        trackerFlag = False
        if detections.any():
            if mot_tracker != None:
                trdata = mot_tracker.update(detections)
                trackerFlag = True
            text_lines = [
                'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
                'FPS: {} fps'.format(round(next(fps_counter))), ]
        if len(objs) != 0:
            return generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines, trdata, trackerFlag)

    result = gstreamer.run_pipeline(user_callback,
                                    src_size=cameras.get_razer_kiyo_resolution(),
                                    appsink_size=inference_size,
                                    trackerName=args.tracker,
                                    videosrc=args.videosrc,
                                    videofmt=args.videofmt)


if __name__ == '__main__':
    main()

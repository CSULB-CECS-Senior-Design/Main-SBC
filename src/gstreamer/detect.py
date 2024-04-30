"""
detect.py
TEST_DATA=../all_models
"""

import collections
from . import common
import numpy as np
import re
import svgwrite


Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])


def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
        return {np.uint8(num): text.strip() for num, text in lines}


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
            x, y, w, h = np.int16(x * inf_w), np.int16(y * inf_h), np.int16(w * inf_w), np.int16(h * inf_h)
            # Subtract boxing offset.
            x, y = x - box_x, y - box_y
            # Scale to source coordinate space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = np.uint8(100 * obj.score)
            label = '{}% {} ID:{}'.format(
                percent, labels.get(obj.id, obj.id), np.uint8(trackID))
            shadow_text(dwg, x, y - 5, label)
            dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                             fill='none', stroke='red', stroke_width='2'))
    else:
        for obj in objs:
            x0, y0, x1, y1, a = list(obj.bbox)
            # Relative coordinates.
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            # Absolute coordinates, input tensor space.
            x, y, w, h = np.int16(x * inf_w), np.int16(y * inf_h), np.int16(w * inf_w), np.int16(h * inf_h)
            # Subtract boxing offset.
            x, y = x - box_x, y - box_y
            # Scale to source coordinate space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = np.uint8(100 * obj.score)
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
            id=np.uint8(category_ids[i]), 
            score=np.float16(scores[i]), 
            bbox=BBox(
                xmin=np.float16(np.maximum(0.0, xmin)), 
                ymin=np.float16(np.maximum(0.0, ymin)), 
                xmax=np.float16(np.minimum(1.0, xmax)), 
                ymax=np.float16(np.minimum(1.0, ymax)), 
                area=np.float16((xmax - xmin) * (ymax - ymin))
            )
        )
    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

def get_proximity(ybbox: float, sections = np.uint8(20)):
    """"Breaks down object in i sections from 0 being furthest to i-1 being closest"""
    return np.uint8(ybbox * 100) % sections

def get_closest_obj(objs: list[Object], min_certainty = np.float16(0.5)) -> Object:
    """Finds the closest object based on y bottom coordinate, then by area, then finally by score.
    Args:
        min_certainty (float, optional): the minimum certainty required for an object to be considered. Defaults to 0.5
    Returns:
        Object: Closest object
    """
    if min_certainty:   # Filter by minimum certainty
        objs = [obj for obj in objs if obj.score > min_certainty]
        
    closest: Object = max(objs, 
                          key=lambda obj: (get_proximity(obj.bbox.ymax), obj.bbox.area, obj.score), 
                          default=None
                          )
    return closest

def is_too_close(obj: Object, xthreshold = np.float16(0.25)) -> bool: # ythreshold = np.float16(0.3)) -> bool:
    """Checks if the given object is too close to the camera.
    Args:
        obj (BBox Object): The object detected in the camera view.
        threshold (float, optional): The threshold for closeness. Defaults to 0.3.
    Returns:
        bool: True if object is too close to the camera, False otherwise.
    """
    # x_too_close = obj.bbox.xmin < xthreshold and 1 - xthreshold < obj.bbox.xmax
    # y_too_close = obj.bbox.ymin < ythreshold and 1 - ythreshold < obj.bbox.ymax
    # return x_too_close or y_too_close
    
    return obj.bbox.xmin < xthreshold and 1 - xthreshold < obj.bbox.xmax

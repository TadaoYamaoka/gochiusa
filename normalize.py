from xml.etree.ElementTree import *
import os
from skimage import io
import math
import sys

infile = sys.argv[1]
outfile = sys.argv[2]

tree = parse(infile)
root = tree.getroot()
images = root.find("images")

def distance(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)

def center(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return [a[0] + dx / 2, a[1] + dy / 2]

for image in list(images):
    img = io.imread(image.get("file"))
    img_width = img.data.shape[1]
    img_height = img.data.shape[0]
    for box in list(image):
        parts = {}
        for part in list(box):
            parts[part.get("name")] = [int(part.get("x")), int(part.get("y"))]

        if "L01" not in parts or "R02" not in parts:
            print(image.get("file"))
            continue

        c = center(parts["L01"], parts["R02"])
        l = distance(c, parts["C02"]) * 1.25
        
        top = int(c[1] - l)
        if top < 0:
            top = 0
        
        left = int(c[0] - l)
        if left < 0:
            lef = 0
        
        bottom = int(c[1] + l)
        if bottom >= img_height:
            bottom = img_height
        
        right = int(c[0] + l)
        if right >= img_width:
            right = img_width
        
        box.set("top", str(top))
        box.set("left", str(left))
        box.set("height", str(bottom - top + 1))
        box.set("width", str(right - left + 1))


tree.write(outfile)

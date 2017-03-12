import sys
import math
import numpy as np
from images2gif import writeGif
from PIL import Image, ImageDraw

assert len(sys.argv) == 3
data = np.loadtxt(sys.argv[1])
assert data.shape[1] == 4
steps = 16
r = 14 
margin = 50 
plotsize = margin*2 + int(math.sqrt(data.shape[0])*(2*r+4.0))
finalsize = plotsize/4 
tile = 1

images = []
color = np.sum(np.sqrt((data[:,0:2] - data[:,2:4])**2), axis=1)
color /= np.max(color)/255
assert color.shape[0] == data.shape[0]
for i,a in enumerate(np.linspace(0.0,1.0,steps)):
  coords = a*data[:,0:2] + (1.0-a)*data[:,2:4]
  image = Image.fromarray(np.full((plotsize*tile, plotsize*tile, 3), 255, dtype=np.uint8))
  draw = ImageDraw.Draw(image)
  for xtile in range(tile):
    for ytile in range(tile):
      for pi,point in enumerate(coords):
        x = (point[0]+xtile)*(plotsize-margin) + margin
        y = (point[1]+ytile)*(plotsize-margin) + margin
        c = int(color[pi])
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(0,c,0))
  image.thumbnail((finalsize*tile,finalsize*tile)) 
  images.append(np.uint8(image)) 

revimages = list(reversed(images))
arrest_frames=8
writeGif(sys.argv[2], 
         [images[0]]*arrest_frames + images + 
         [revimages[0]]*arrest_frames + revimages, 
         duration=0.06, dither=0)

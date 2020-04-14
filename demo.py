import cv2
from color_transfer import color_transfer
from weightedGE import weightedGE, im2double

source = 'data/example3/source.jpg'
target = 'data/example3/target.jpg'
output = 'data/example3/output.jpg'

Is = im2double(cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB))
It = im2double(cv2.cvtColor(cv2.imread(target), cv2.COLOR_BGR2RGB))
Io = color_transfer(Is, It)
cv2.imwrite(output, Io)




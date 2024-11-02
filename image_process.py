from PIL import Image
import pylab

import rof

im = pylab.array(Image.open('empire.jpg').convert('L'))
U,T = rof.denoise(im,im)

pylab.figure()
pylab.gray()
pylab.imshow(U)
pylab.axis('equal')
pylab.axis('off')
pylab.show()
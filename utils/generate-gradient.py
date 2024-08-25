import math
from PIL import Image, ImageDraw

if __name__ == '__main__':
    height = 256
    image = Image.new('L', (256, height))
    draw = ImageDraw.Draw(image)
    for level in range(256):
        print(round(level))
        draw.rectangle((0, level, height, level + 1), level)

    image.save('gradient.png')

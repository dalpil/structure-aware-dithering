
from PIL import Image, ImageDraw

if __name__ == '__main__':
    desired_width = 800
    height = 100

    levels = []
    divisor = 2
    for index in range(divisor + 1):
        levels.append(int((1 / divisor) * index * 127))

    levels += [128 + x for x in levels[1:]]

    # Levels near the extreme ends tend to introduce lots of artifacts / worms
    levels = [2] + levels[1:-1] + [253]

    step_width = desired_width // len(levels)
    height = step_width

    image = Image.new('L', (step_width*len(levels), height))
    draw = ImageDraw.Draw(image)
    for index, level in enumerate(levels):
        draw.rectangle((index * step_width, 0, index * step_width + step_width, height), level)

    image.save('levels.png')

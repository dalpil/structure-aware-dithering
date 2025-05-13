#!/usr/bin/env python3

import heapq
import math
import random

import click
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
import numba
import numpy as np
import scipy
import skimage
from skimage.filters import gaussian


DITHER_KERNELS = {
    'atkinson': (
        (1, 0, 1 / 8),
        (2, 0, 1 / 8),
        (-1, 1, 1 / 8),
        (0, 1, 1 / 8),
        (1, 1, 1 / 8),
        (0, 2, 1 / 8),
    ),

    'floyd-steinberg': (
        (1, 0, 7 / 16),
        (-1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (1, 1, 1 / 16),
    ),

    'jarvis-judice-ninke': (
        (1, 0, 7 / 48),
        (2, 0, 5 / 48),
        (-2, 1, 3 / 48),
        (-1, 1, 5 / 48),
        (0, 1, 7 / 48),
        (1, 1, 5 / 48),
        (2, 1, 3 / 48),
        (-2, 2, 1 / 48),
        (-1, 2, 3 / 48),
        (0, 2, 5 / 48),
        (1, 2, 3 / 48),
        (2, 2, 1 / 48),
    ),

    'stucki': (
        (1, 0, 8 / 42),
        (2, 0, 4 / 42),
        (-2, 1, 2 / 42),
        (-1, 1, 4 / 42),
        (0, 1, 8 / 42),
        (1, 1, 4 / 42),
        (2, 1, 2 / 42),
        (-2, 2, 1 / 42),
        (-1, 2, 2 / 42),
        (0, 2, 4 / 42),
        (1, 2, 2 / 42),
        (2, 2, 1 / 42),
    ),

    'burkes': (
        (1, 0, 8 / 32),
        (2, 0, 4 / 32),
        (-2, 1, 2 / 32),
        (-1, 1, 4 / 32),
        (0, 1, 8 / 32),
        (1, 1, 4 / 32),
        (2, 1, 2 / 32),
    ),

    'sierra3': (
        (1, 0, 5 / 32),
        (2, 0, 3 / 32),
        (-2, 1, 2 / 32),
        (-1, 1, 4 / 32),
        (0, 1, 5 / 32),
        (1, 1, 4 / 32),
        (2, 1, 2 / 32),
        (-1, 2, 2 / 32),
        (0, 2, 3 / 32),
        (1, 2, 2 / 32),
    ),

    'sierra2': (
        (1, 0, 4 / 16),
        (2, 0, 3 / 16),
        (-2, 1, 1 / 16),
        (-1, 1, 2 / 16),
        (0, 1, 3 / 16),
        (1, 1, 2 / 16),
        (2, 1, 1 / 16),
    ),

    'sierra-lite': (
        (1, 0, 2 / 4),
        (-1, 1, 1 / 4),
        (0, 1, 1 / 4),
    ),

    # Shiau-Fan set out to reduce worm artifacts
    'shiau-fan1': (
        (1, 0, 1 / 2),
        (-2, 0, 1 / 8),
        (-1, 0, 1 / 8),
        (0, 1, 1 / 4),
    ),

    'shiau-fan2': (
        (1, 0, 1 / 2),
        (-3, 1, 1 / 16),
        (-2, 1, 1 / 16),
        (-1, 1, 1 / 8),
        (0, 1, 1 / 4),
    ),

    # https://doi.org/10.1117/12.271597
    'wong-allebach': (
        (1, 0, 0.2911),

        (-1, 1, 0.1373),
        (0, 1, 0.3457),
        (1, 1, 0.2258)
    ),

    # https://doi.org/10.1117/12.2180540
    'fedoseev': (
        (1, 0, 0.5423),
        (2, 0, 0.0533),

        (-2, 1, 0.0246),
        (-1, 1, 0.2191),
        (0, 1, 0.4715),
        (1, 1, -0.0023),
        (2, 1, -0.1241),

        (-2, 2, -0.0065),
        (-1, 2, -0.0692),
        (0, 2, 0.0168),
        (1, 2, -0.0952),
        (2, 2, -0.0304),
    ),

    'fedoseev2': (
        (1, 0, 0.4364),
        (0, 1, 0.5636),
    ),

    'fedoseev3': (
        (1, 0, 0.4473),
        (-1, 1, 0.1654),
        (0, 1, 0.3872),
    ),

    'fedoseev4': (
        (1, 0, 0.5221),
        (-1, 1, 0.1854),
        (0, 1, 0.4689),
        (1, 2, -0.1763),
    ),
}


# This function has been lifted from https://github.com/laszlokorte/blue-noise with some minor adjustments from my side
def generate_bluenoise(shape, initial_ratio=0.1, sigma=1.5):
    ranks = np.zeros(shape)

    initial_white_noise = np.random.random_sample(shape)
    placed_pixels = initial_white_noise <= initial_ratio
    count_placed = np.sum(placed_pixels)
    count_remaining = placed_pixels.size - count_placed

    prev_swap = None

    # Phase 1: Place intial
    while True:
        blurred = gaussian(placed_pixels, sigma, mode='wrap')
        densest = (blurred * placed_pixels).argmax()
        voidest = (blurred + placed_pixels).argmin()

        if prev_swap == (voidest, densest):
            break

        densest_coord = np.unravel_index(densest, shape)
        voidest_coord = np.unravel_index(voidest, shape)

        placed_pixels[densest_coord] = False
        placed_pixels[voidest_coord] = True

        prev_swap = (densest, voidest)

    # Phase 2: Rank pixels by density
    placed_but_not_ranked = placed_pixels.copy()

    for rank in range(count_placed, 0, -1):
        blurred = gaussian(placed_pixels, sigma, mode='wrap')

        densest = (blurred * placed_but_not_ranked).argmax()
        densest_coord = np.unravel_index(densest, shape)

        placed_but_not_ranked[densest_coord] = False
        ranks[densest_coord] = rank

    # Phase 3: Fill up remaining pixels from the sparsest areas
    for rank in range(count_remaining):
        blurred = gaussian(placed_pixels, sigma, mode='wrap')

        voidest = (blurred + placed_pixels).argmin()
        voidest_coord = np.unravel_index(voidest, shape)

        placed_pixels[voidest_coord] = True
        ranks[voidest_coord] = count_placed + rank

    return ranks


@numba.jit(nopython=True)
def dither_threshold(input, threshold, noise):
    output = np.zeros_like(input)

    height, width = input.shape
    for y in range(height):
        for x in range(width):
            output[y, x] = 0.0 if input[y, x] < threshold + noise * np.random.uniform(-1.0, 1.0) else 1.0

    return output


@numba.jit(nopython=True)
def dither_whitenoise(input):
    output = np.zeros_like(input)

    height, width = input.shape
    for y in range(height):
        for x in range(width):
            output[y, x] = 0.0 if input[y, x] < np.random.uniform(0.0, 1.0) else 1.0

    return output


def dither_bluenoise(input, noise_shape):
    bnr = generate_bluenoise(noise_shape)
    bn = (bnr - bnr.min()) / (bnr.max() - bnr.min())

    bnh, bnw = noise_shape
    output = np.zeros_like(input)

    height, width = input.shape
    for y in range(height):
        for x in range(width):
            output[y, x] = 0.0 if input[y, x] < bn[y % bnh, x % bnw] else 1.0

    return output


# Faster without JIT
# @numba.jit(nopython=True)
def dither_bayer(input, bayer_matrix):
    input = input.copy()
    output = np.zeros_like(input)
    height, width = input.shape
    mheight, mwidth = bayer_matrix.shape
    for y in range(height):
        for x in range(width):
            if input[y, x] == 0.0 or input[y, x] == 1.0:
                continue

            input[y][x] = 0.0 if input[y][x] < bayer_matrix[y % mheight][x % mwidth] else 1.0

    return input


@numba.jit(nopython=True)
def dither_classic(original, diff_map, serpentine, k=0.0, noise_multiplier=0.0):
    input = original.copy()
    output = np.zeros_like(input)

    direction = 1
    height, width = input.shape

    for y in range(height):
        for x in range(0, width, direction) if direction > 0 else range(width - 1, -1, direction):
            old_pixel = input[y, x]
            new_pixel = 0.0 if old_pixel + (k * (original[y, x] - 0.5)) < 0.5 + noise_multiplier * np.random.uniform(-0.5, 0.5) else 1.0
            quantization_error = old_pixel - new_pixel
            output[y, x] = new_pixel

            for dx, dy, diffusion_coefficient in diff_map:
                # Reverse the kernel if we are going right to left
                if direction < 0:
                    dx *= -1

                xn, yn = x + dx, y + dy

                if (0 <= xn < width) and (0 <= yn < height):
                    # Some kernels use negative coefficients, so we cannot clamp this value between 0.0-1.0
                    input[yn, xn] = input[yn, xn] + (quantization_error * diffusion_coefficient)

            if serpentine and ((direction > 0 and x >= (width - 1)) or (direction < 0 and x <= 0)):
                direction *= -1

    return output


@numba.jit(nopython=True)
def dither_classic_modulated(input, diff_map):
    input = input.copy()
    output = np.zeros_like(input)

    height, width = input.shape

    for y in range(height):
        for x in range(0, width):
            old_pixel = input[y, x]
            new_pixel = 0.0 if old_pixel < 0.5 + np.random.uniform(low=-0.15, high=0.15) else 1.0
            quantization_error = old_pixel - new_pixel
            output[y, x] = new_pixel

            for dx, dy, diffusion_coefficient in diff_map:
                xn, yn = x + dx, y + dy

                if (0 <= xn < width) and (0 <= yn < height):
                    new_value = input[yn, xn] + (quantization_error * diffusion_coefficient)
                    input[yn, xn] = 0.0 if new_value < 0.0 else 1.0 if new_value > 1.0 else new_value

    return output


# A simple and efficient error-diffusion algorithm
# DOI: https://doi.org/10.1145/383259.383326
# PDF: https://perso.liris.cnrs.fr/victor.ostromoukhov/publications/pdf/SIGGRAPH01_varcoeffED.pdf
# Slower with JIT
# @numba.jit(nopython=True)
def dither_ostromoukhov(input):
    input = input.copy()
    output = np.zeros_like(input)

    coefficients = [
        (13,     0,     5,    18),
        (13,     0,     5,    18),
        (21,     0,    10,    31),
        (7,     0,     4,    11),
        (8,     0,     5,    13),
        (47,     3,    28,    78),
        (23,     3,    13,    39),
        (15,     3,     8,    26),
        (22,     6,    11,    39),
        (43,    15,    20,    78),
        (7,     3,     3,    13),
        (501,   224,   211,   936),
        (249,   116,   103,   468),
        (165,    80,    67,   312),
        (123,    62,    49,   234),
        (489,   256,   191,   936),
        (81,    44,    31,   156),
        (483,   272,   181,   936),
        (60,    35,    22,   117),
        (53,    32,    19,   104),
        (237,   148,    83,   468),
        (471,   304,   161,   936),
        (3,     2,     1,     6),
        (459,   304,   161,   924),
        (38,    25,    14,    77),
        (453,   296,   175,   924),
        (225,   146,    91,   462),
        (149,    96,    63,   308),
        (111,    71,    49,   231),
        (63,    40,    29,   132),
        (73,    46,    35,   154),
        (435,   272,   217,   924),
        (108,    67,    56,   231),
        (13,     8,     7,    28),
        (213,   130,   119,   462),
        (423,   256,   245,   924),
        (5,     3,     3,    11),
        (281,   173,   162,   616),
        (141,    89,    78,   308),
        (283,   183,   150,   616),
        (71,    47,    36,   154),
        (285,   193,   138,   616),
        (13,     9,     6,    28),
        (41,    29,    18,    88),
        (36,    26,    15,    77),
        (289,   213,   114,   616),
        (145,   109,    54,   308),
        (291,   223,   102,   616),
        (73,    57,    24,   154),
        (293,   233,    90,   616),
        (21,    17,     6,    44),
        (295,   243,    78,   616),
        (37,    31,     9,    77),
        (27,    23,     6,    56),
        (149,   129,    30,   308),
        (299,   263,    54,   616),
        (75,    67,    12,   154),
        (43,    39,     6,    88),
        (151,   139,    18,   308),
        (303,   283,    30,   616),
        (38,    36,     3,    77),
        (305,   293,    18,   616),
        (153,   149,     6,   308),
        (307,   303,     6,   616),
        (1,     1,     0,     2),
        (101,   105,     2,   208),
        (49,    53,     2,   104),
        (95,   107,     6,   208),
        (23,    27,     2,    52),
        (89,   109,    10,   208),
        (43,    55,     6,   104),
        (83,   111,    14,   208),
        (5,     7,     1,    13),
        (172,   181,    37,   390),
        (97,    76,    22,   195),
        (72,    41,    17,   130),
        (119,    47,    29,   195),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (65,    18,    17,   100),
        (95,    29,    26,   150),
        (185,    62,    53,   300),
        (30,    11,     9,    50),
        (35,    14,    11,    60),
        (85,    37,    28,   150),
        (55,    26,    19,   100),
        (80,    41,    29,   150),
        (155,    86,    59,   300),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (305,   176,   119,   600),
        (155,    86,    59,   300),
        (105,    56,    39,   200),
        (80,    41,    29,   150),
        (65,    32,    23,   120),
        (55,    26,    19,   100),
        (335,   152,   113,   600),
        (85,    37,    28,   150),
        (115,    48,    37,   200),
        (35,    14,    11,    60),
        (355,   136,   109,   600),
        (30,    11,     9,    50),
        (365,   128,   107,   600),
        (185,    62,    53,   300),
        (25,     8,     7,    40),
        (95,    29,    26,   150),
        (385,   112,   103,   600),
        (65,    18,    17,   100),
        (395,   104,   101,   600),
        (4,     1,     1,     6),
    ]

    coefficients = [[x / 255 for x in t] for t in coefficients]

    coefficients_reversed = coefficients.copy()
    coefficients_reversed.reverse()
    coefficients = coefficients + coefficients_reversed

    left_to_right = True
    height, width = input.shape

    for y in range(height):
        if left_to_right:
            r = range(0, width, 1)
        else:
            r = range(width - 1, -1, -1)

        for x in r:
            old_pixel = input[y, x]
            new_pixel = 0.0 if old_pixel <= 0.5 else 1.0
            quantization_error = old_pixel - new_pixel
            output[y, x] = new_pixel

            directions = ((1, 0), (-1, 1), (0, 1))
            tmp = coefficients[int(old_pixel * 255.0)]
            intensity_coefficients = tmp[0:3]
            intensity_coefficients_sum = tmp[3]
            for (dx, dy), pixel_coefficient in zip(directions, intensity_coefficients):
                if not left_to_right:
                    dx *= -1

                xn, yn = x + dx, y + dy

                if (0 <= xn < width) and (0 <= yn < height):
                    new_value = input[yn, xn] + pixel_coefficient * quantization_error / intensity_coefficients_sum
                    input[yn, xn] = 0.0 if new_value <= 0.0 else 1.0 if new_value >= 1.0 else new_value

        if left_to_right and x >= (width - 1):
            left_to_right = False

        if not left_to_right and x <= 0:
            left_to_right = True

    return output


# Improving mid-tone quality of variable-coefficient error diffusion using threshold modulation
# DOI: https://doi.org/10.1145/882262.882289
# PDF: https://www.wict.pku.edu.cn/GraphicsLab/docs/20200416110604416469.pdf
@numba.jit(nopython=True)
def dither_zhoufang(input, coefficients, modulator):
    input = input.copy()
    output = np.zeros_like(input)

    modulate = lambda r, v: 0.5 + (r % 0.5) * modulator[int(v * 255.0)]

    left_to_right = True
    height, width = input.shape

    for y in range(height):
        if left_to_right:
            r = range(0, width, 1)
        else:
            r = range(width - 1, -1, -1)

        for x in r:
            old_pixel = input[y, x]
            intensity_coefficients = coefficients[int(old_pixel * 255.0)]

            rnd = np.random.uniform(0.0, 1.0)
            new_pixel = 0.0 if old_pixel < modulate(rnd, old_pixel) else 1.0

            quantization_error = old_pixel - new_pixel
            output[y, x] = new_pixel

            directions = ((1, 0), (-1, 1), (0, 1))
            for (dx, dy), pixel_coefficient in zip(directions, intensity_coefficients):
                if not left_to_right:
                    dx *= -1

                xn, yn = x + dx, y + dy

                if (0 <= xn < width) and (0 <= yn < height):
                    new_value = input[yn, xn] + pixel_coefficient * quantization_error
                    input[yn, xn] = 0.0 if new_value < 0.0 else 1.0 if new_value > 1.0 else new_value

            if left_to_right and x >= (width - 1):
                left_to_right = False

            if not left_to_right and x <= 0:
                left_to_right = True

    return output


# Rapidly Creating Structure-aware Halftoning with Improved Error Diffusion
# DOI: https://doi.org/10.1109/CISP.2009.5303919
@numba.jit(nopython=True)
def dither_zhangpang(input, coefficients, modulator, c):
    input = input.copy() * 255
    original = input.copy()
    output = np.zeros_like(input)

    modulate = lambda r, v: 0.5 + (r % 0.5) * modulator[int(v)][0]

    left_to_right = True
    height, width = input.shape

    w = np.array([[0.1035, 0.1465, 0.1035],
                  [0.1465, 0.0,    0.1465],
                  [0.1035, 0.1465, 0.1035]])

    for y in range(height):
        r = range(0, width, 1) if left_to_right else range(width - 1, -1, -1)

        for x in r:
            itf = 0

            if (x > 0 and x < width - 1) and (y > 0 and y < height - 1):
                average_luminosity = np.mean(original[y-1:y+2, x-1:x+2])
                visual_perception_error = original[y-1:y+2, x-1:x+2] - average_luminosity

                spatial_variation = np.sum(w * np.abs(visual_perception_error))
                spatial_activity_measure = spatial_variation * (original[y, x] - average_luminosity)
                itf = c * average_luminosity * spatial_activity_measure

                # NOTE: This is not in the paper, but I'm getting awful results without it.
                itf = max(-127, min(127, itf))

            old_pixel = input[y, x]
            intensity_coefficients = coefficients[int(old_pixel)]

            rnd = np.random.uniform(0.0, 1.0)
            new_pixel = 0 if old_pixel + itf < (255 * modulate(rnd, old_pixel)) else 255

            quantization_error = old_pixel - new_pixel
            output[y, x] = int(new_pixel)

            directions = ((1, 0), (-1, 1), (0, 1))
            for (dx, dy), pixel_coefficient in zip(directions, intensity_coefficients):
                if not left_to_right:
                    dx *= -1

                xn, yn = x + dx, y + dy

                if (0 <= xn < width) and (0 <= yn < height):
                    new_value = input[yn, xn] + pixel_coefficient * quantization_error
                    input[yn, xn] = max(0, min(255, new_value))

            if left_to_right and x >= (width - 1):
                left_to_right = False

            if not left_to_right and x <= 0:
                left_to_right = True

    return output / 255


# Structure-aware error-diffusion approach using entropy-constrained threshold modulation
# https://doi.org/10.1007/s00371-013-0895-0
@numba.jit(nopython=True)
def dither_entropy_constrained(input, input_highpass, c, diff_map, serpentine):
    input = input.copy()
    output = np.zeros_like(input)

    entropy = lambda i: 0.0 if i <= 0.0 else 0.0 if i >= 1.0 else -i * math.log(i) - (1.0 - i) * math.log(1.0 - i)

    direction = 1
    height, width = input.shape

    for y in range(height):
        for x in range(0, width, direction) if direction > 0 else range(width - 1, -1, direction):
            old_pixel = input[y, x]
            new_pixel = 0.0 if old_pixel < (0.5 + (c * entropy(old_pixel) * input_highpass[y, x])) else 1.0

            quantization_error = old_pixel - new_pixel
            output[y, x] = new_pixel

            for dx, dy, diffusion_coefficient in diff_map:
                # Reverse the kernel if we are going right to left
                if direction < 0:
                    dx *= -1

                xn, yn = x + dx, y + dy

                if (0 <= xn < width) and (0 <= yn < height):
                    new_value = input[yn, xn] + (quantization_error * diffusion_coefficient)
                    input[yn, xn] = 0.0 if new_value < 0.0 else 1.0 if new_value > 1.0 else new_value

            if serpentine and ((direction > 0 and x >= (width - 1)) or (direction < 0 and x <= 0)):
                direction *= -1

    return output


# Structure-aware error-diffusion approach using entropy-constrained threshold modulation
# https://doi.org/10.1007/s00371-013-0895-0
@numba.jit(nopython=True)
def dither_entropy_constrained_ostromoukhov(input, input_highpass, c):
    input = input.copy()
    output = np.zeros_like(input)

    coefficients = [
        (13,     0,     5,    18),
        (13,     0,     5,    18),
        (21,     0,    10,    31),
        (7,     0,     4,    11),
        (8,     0,     5,    13),
        (47,     3,    28,    78),
        (23,     3,    13,    39),
        (15,     3,     8,    26),
        (22,     6,    11,    39),
        (43,    15,    20,    78),
        (7,     3,     3,    13),
        (501,   224,   211,   936),
        (249,   116,   103,   468),
        (165,    80,    67,   312),
        (123,    62,    49,   234),
        (489,   256,   191,   936),
        (81,    44,    31,   156),
        (483,   272,   181,   936),
        (60,    35,    22,   117),
        (53,    32,    19,   104),
        (237,   148,    83,   468),
        (471,   304,   161,   936),
        (3,     2,     1,     6),
        (459,   304,   161,   924),
        (38,    25,    14,    77),
        (453,   296,   175,   924),
        (225,   146,    91,   462),
        (149,    96,    63,   308),
        (111,    71,    49,   231),
        (63,    40,    29,   132),
        (73,    46,    35,   154),
        (435,   272,   217,   924),
        (108,    67,    56,   231),
        (13,     8,     7,    28),
        (213,   130,   119,   462),
        (423,   256,   245,   924),
        (5,     3,     3,    11),
        (281,   173,   162,   616),
        (141,    89,    78,   308),
        (283,   183,   150,   616),
        (71,    47,    36,   154),
        (285,   193,   138,   616),
        (13,     9,     6,    28),
        (41,    29,    18,    88),
        (36,    26,    15,    77),
        (289,   213,   114,   616),
        (145,   109,    54,   308),
        (291,   223,   102,   616),
        (73,    57,    24,   154),
        (293,   233,    90,   616),
        (21,    17,     6,    44),
        (295,   243,    78,   616),
        (37,    31,     9,    77),
        (27,    23,     6,    56),
        (149,   129,    30,   308),
        (299,   263,    54,   616),
        (75,    67,    12,   154),
        (43,    39,     6,    88),
        (151,   139,    18,   308),
        (303,   283,    30,   616),
        (38,    36,     3,    77),
        (305,   293,    18,   616),
        (153,   149,     6,   308),
        (307,   303,     6,   616),
        (1,     1,     0,     2),
        (101,   105,     2,   208),
        (49,    53,     2,   104),
        (95,   107,     6,   208),
        (23,    27,     2,    52),
        (89,   109,    10,   208),
        (43,    55,     6,   104),
        (83,   111,    14,   208),
        (5,     7,     1,    13),
        (172,   181,    37,   390),
        (97,    76,    22,   195),
        (72,    41,    17,   130),
        (119,    47,    29,   195),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (4,     1,     1,     6),
        (65,    18,    17,   100),
        (95,    29,    26,   150),
        (185,    62,    53,   300),
        (30,    11,     9,    50),
        (35,    14,    11,    60),
        (85,    37,    28,   150),
        (55,    26,    19,   100),
        (80,    41,    29,   150),
        (155,    86,    59,   300),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (5,     3,     2,    10),
        (305,   176,   119,   600),
        (155,    86,    59,   300),
        (105,    56,    39,   200),
        (80,    41,    29,   150),
        (65,    32,    23,   120),
        (55,    26,    19,   100),
        (335,   152,   113,   600),
        (85,    37,    28,   150),
        (115,    48,    37,   200),
        (35,    14,    11,    60),
        (355,   136,   109,   600),
        (30,    11,     9,    50),
        (365,   128,   107,   600),
        (185,    62,    53,   300),
        (25,     8,     7,    40),
        (95,    29,    26,   150),
        (385,   112,   103,   600),
        (65,    18,    17,   100),
        (395,   104,   101,   600),
        (4,     1,     1,     6),
    ]

    coefficients_reversed = coefficients.copy()
    coefficients_reversed.reverse()
    coefficients = coefficients + coefficients_reversed

    entropy = lambda i: 0.0 if i <= 0.0 else 0.0 if i >= 1.0 else -i * math.log(i) - (1.0 - i) * math.log(1.0 - i)

    left_to_right = True
    height, width = input.shape

    for y in range(height):
        if left_to_right:
            r = range(0, width, 1)
        else:
            r = range(width - 1, -1, -1)

        for x in r:
            old_pixel = input[y, x]
            new_pixel = 0.0 if old_pixel < (0.5 + (c * entropy(old_pixel) * input_highpass[y, x])) else 1.0
            quantization_error = old_pixel - new_pixel
            output[y, x] = new_pixel

            directions = ((1, 0), (-1, 1), (0, 1))
            tmp = [x / 255.0 for x in coefficients[int(old_pixel * 255.0)]]
            intensity_coefficients = tmp[0:3]
            intensity_coefficients_sum = tmp[3]
            for (dx, dy), pixel_coefficient in zip(directions, intensity_coefficients):
                if not left_to_right:
                    dx *= -1

                xn, yn = x + dx, y + dy

                if (0 <= xn < width) and (0 <= yn < height):
                    new_value = input[yn, xn] + pixel_coefficient * quantization_error / intensity_coefficients_sum
                    input[yn, xn] = 0.0 if new_value < 0.0 else 1.0 if new_value > 1.0 else new_value

        if left_to_right and x >= (width - 1):
            left_to_right = False

        if not left_to_right and x <= 0:
            left_to_right = True

    return output


# Structure-aware error-diffusion approach using entropy-constrained threshold modulation
# https://doi.org/10.1007/s00371-013-0895-0
@numba.jit(nopython=True)
def dither_entropy_constrained_zhoufang(input, coefficients, modulator, input_highpass, c):
    input = input.copy()
    output = np.zeros_like(input)

    modulate = lambda r, v: 0.5 + (r % 0.5) * modulator[int(v * 255.0)]
    entropy = lambda i: 0.0 if i <= 0.0 else 0.0 if i >= 1.0 else -i * math.log(i) - (1.0 - i) * math.log(1.0 - i)

    left_to_right = True
    height, width = input.shape

    for y in range(height):
        if left_to_right:
            r = range(0, width, 1)
        else:
            r = range(width - 1, -1, -1)

        for x in r:
            old_pixel = input[y, x]
            intensity_coefficients = coefficients[int(old_pixel * 255.0)]

            new_pixel = 0.0 if old_pixel < modulate(np.random.uniform(0.0, 1.0), old_pixel) + (c * entropy(old_pixel) * input_highpass[y, x]) else 1.0

            quantization_error = old_pixel - new_pixel
            output[y, x] = new_pixel

            directions = ((1, 0), (-1, 1), (0, 1))
            for (dx, dy), pixel_coefficient in zip(directions, intensity_coefficients):
                if not left_to_right:
                    dx *= -1

                xn, yn = x + dx, y + dy

                if (0 <= xn < width) and (0 <= yn < height):
                    new_value = input[yn, xn] + pixel_coefficient * quantization_error
                    input[yn, xn] = 0.0 if new_value < 0.0 else 1.0 if new_value > 1.0 else new_value

            if left_to_right and x >= (width - 1):
                left_to_right = False

            if not left_to_right and x <= 0:
                left_to_right = True

    return output


# Laplacian based structure-aware error diffusion
# DOI: https://doi.org/10.1109/ICIP.2010.5651243
# PDF: https://www.researchgate.net/profile/K-K-Kong/publication/224200181_Laplacian_based_structure-aware_error_diffusion/links/57a7c92a08aefe6167bc6b59/Laplacian-based-structure-aware-error-diffusion.pdf
# Faster without JIT
# @numba.jit(nopython=True)
def dither_laplacian(input, diff_map, serpentine, scale_factor):
    def window_stdev(arr, radius):
        c1 = scipy.ndimage.uniform_filter(arr, radius*2, mode='reflect', origin=-radius)
        c2 = scipy.ndimage.uniform_filter(arr*arr, radius*2, mode='reflect', origin=-radius)
        return (np.clip(c2 - c1*c1, 0.0, 1.0)**.5)

    input = input.copy()
    output = np.zeros_like(input)

    lm = scipy.ndimage.laplace(input)
    np.clip(lm, -0.5, 0.5, out=lm)

    global_stddev = np.std(input)

    stddev = window_stdev(input, 5)
    l_max = scipy.ndimage.maximum_filter(stddev, size=10)
    l_min = scipy.ndimage.minimum_filter(stddev, size=10)

    direction = 1
    height, width = input.shape

    for y in range(height):
        for x in range(0, width, direction) if direction > 0 else range(width - 1, -1, direction):
            local_stddev = stddev[y, x]
            local_max = l_max[y, x]
            local_min = l_min[y, x]

            K = 0.0
            if local_max - local_min != 0.0:
                K = scale_factor / global_stddev * (abs(local_stddev - local_max) / (local_max - local_min)) + scale_factor

            N = np.random.normal(0, 0.1)
            T = K * lm[y, x] + N

            old_pixel = input[y, x]
            new_pixel = 0.0 if old_pixel < 0.5 + T else 1.0
            quantization_error = old_pixel - new_pixel
            output[y, x] = new_pixel

            for dx, dy, diffusion_coefficient in diff_map:
                # Reverse the kernel if we are going right to left
                if direction < 0:
                    dx *= -1

                xn, yn = x + dx, y + dy

                if (0 <= xn < width) and (0 <= yn < height):
                    new_value = input[yn, xn] + (quantization_error * diffusion_coefficient)
                    input[yn, xn] = 0.0 if new_value < 0.0 else 1.0 if new_value > 1.0 else new_value

            if serpentine and ((direction > 0 and x >= (width - 1)) or (direction < 0 and x <= 0)):
                direction *= -1

    return output


# Contrast-aware Halftoning
# DOI: https://doi.org/10.1111/j.1467-8659.2009.01596.x
# PDF: https://people.scs.carleton.ca/~mould/papers/ContrastAwareHalftoning-final.pdf
def dither_contrast_aware_basic(input, mask_size, k_parameter):
    @numba.jit(nopython=True)
    def circular(x, y, r, xw, yw):
        for py in range(y - r, y + r + 1):
            for px in range(x - r, x + r + 1):
                # Skip the center pixel
                if (px, py) == (x, y):
                    continue

                # Distance of this pixel from centre, squared
                ds = (px - x) ** 2 + (py - y) ** 2

                # Yield pixel if within circle and within bounds
                if ds <= r ** 2 and 0 <= px < xw and 0 <= py < yw:
                    yield px, py

    input = input.copy()
    output = np.zeros_like(input)

    visited_pixels = set()
    residual_error = 0
    height, width = input.shape

    for y in range(0, height):
        for x in range(0, width):
            intensity = input[y, x] + residual_error
            residual_error = 0

            threshold = 0.5
            if intensity < threshold:
                error = intensity
                output[y, x] = 0.0
            else:
                error = intensity - 1.0
                output[y, x] = 1.0

            # Calculate the weights of the surrounding pixels in a circular mask
            weights = {}
            for mx, my in circular(x, y, mask_size // 2, width, height):
                # Skip already visited pixels
                if (mx, my) in visited_pixels:
                    continue

                mask_intensity = input[my, mx]
                mask_distance_from_center = math.sqrt((mx - x) ** 2 + (my - y) ** 2)

                if error > 0.0:
                    mask_weight = mask_intensity / (mask_distance_from_center ** k_parameter)
                else:
                    mask_weight = (1.0 - mask_intensity) / (mask_distance_from_center ** k_parameter)

                weights[(mx, my)] = mask_weight

            # Calculate the total weight of the mask
            total_weight = sum(weights.values())

            # Only distribute the error if the total weight is greater than 0
            if total_weight != 0.0:

                # Calculate the normalized weight of each pixel
                normalized_weights = {k: v / total_weight for k, v in weights.items()}

                # Calculate the new intensity of each pixel by distributing the error
                for pixel, weight in normalized_weights.items():
                    new_intensity = input[pixel[1], pixel[0]] + error * weight

                    if new_intensity > 1.0:
                        residual_error += new_intensity - 1.0
                    elif new_intensity < 0.0:
                        residual_error += new_intensity

                    input[pixel[1], pixel[0]] = max(0.0, min(1.0, new_intensity))

            visited_pixels.add((x, y))

    return output


# Contrast-aware Halftoning
# DOI: https://doi.org/10.1111/j.1467-8659.2009.01596.x
# PDF: https://people.scs.carleton.ca/~mould/papers/ContrastAwareHalftoning-final.pdf
def dither_contrast_aware_variant(input, mask_size, k_parameter):
    @numba.jit(nopython=True)
    def circular(x, y, r, xw, yw):
        for py in range(y - r, y + r + 1):
            for px in range(x - r, x + r + 1):
                # Skip the center pixel
                if (px, py) == (x, y):
                    continue

                # Distance of this pixel from centre, squared
                ds = (px - x) ** 2 + (py - y) ** 2

                # Yield pixel if within circle and within bounds
                if ds <= r ** 2 and 0 <= px < xw and 0 <= py < yw:
                    yield px, py

    input = input.copy()
    output = np.zeros_like(input)
    visited_pixels = np.zeros_like(input, dtype=np.uint8)

    residual_error = 0
    height, width = input.shape

    # Build our pixel intensity heap, with random tiebreakers
    heap = []
    num_pixels = height * width
    tiebreakers = random.sample(range(0, num_pixels), k=num_pixels)

    for y in range(height):
        for x in range(width):
            intensity = input[y, x]

            # Distance to black OR white
            heapq.heappush(heap, (0, int(min(intensity * 255, (1.0 - intensity) * 255)), tiebreakers.pop(), (x, y)))

    while heap:
        priority, intensity_distance, tiebreaker, (x, y) = heapq.heappop(heap)
        intensity = input[y, x]

        # Verify intensity distance, skip if stale
        if visited_pixels[y, x] or intensity_distance != int(min(intensity * 255, (1.0 - intensity) * 255)):
            continue

        intensity += residual_error
        residual_error = 0

        if intensity < 0.5:
            error = intensity
            output[y, x] = 0.0
        else:
            error = intensity - 1.0
            output[y, x] = 1.0

        visited_pixels[y, x] = 1

        if int(error * 255) == 0:
            continue

        # Calculate the weights of the surrounding pixels in a circular mask
        weights = {}
        for mx, my in circular(x, y, mask_size // 2, width, height):
            # Skip already visited pixels
            if visited_pixels[my, mx]:
                continue

            mask_intensity = input[my, mx]
            mask_distance_from_center = math.sqrt((mx - x) ** 2 + (my - y) ** 2)

            if error > 0.0:
                mask_weight = mask_intensity / (mask_distance_from_center ** k_parameter)
            else:
                mask_weight = (1.0 - mask_intensity) / (mask_distance_from_center ** k_parameter)

            weights[(mx, my)] = mask_weight

        # Calculate the total weight of the mask
        total_weight = sum(weights.values())

        # Only distribute the error if the total weight is greater than 0
        if int(total_weight * 255) != 0:

            # Calculate the normalized weight of each pixel
            normalized_weights = {k: v / total_weight for k, v in weights.items()}

            # Calculate the new intensity of each pixel by distributing the error
            for pixel, weight in normalized_weights.items():
                new_intensity = input[pixel[1], pixel[0]] + error * weight

                if new_intensity > 1.0:
                    residual_error += new_intensity - 1.0
                elif new_intensity < 0.0:
                    residual_error += new_intensity

                new_intensity = max(0.0, min(1.0, new_intensity))
                input[pixel[1], pixel[0]] = new_intensity
                heapq.heappush(heap, (priority + 1, int(min(new_intensity * 255, (1.0-new_intensity) * 255)), tiebreaker, pixel))

    return output


# An improved error diffusion algorithm based on visual difference
# DOI: https://doi.org/10.1109/ICIP.2014.7025530
# PDF: https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2014/Papers/1569901225.pdf
def dither_visual_difference(input, num_iterations, uc, q, ramf_enabled, ramf_min, ramf_max):
    @numba.jit(nopython=True)
    def ramf(input, window_size, max_window_size):
        output = input.copy()

        pad_size = window_size // 2
        for i in range(pad_size, input.shape[0] - pad_size):
            for j in range(pad_size, input.shape[1] - pad_size):
                window = input[i - pad_size : i + pad_size + 1, j - pad_size : j + pad_size + 1]
                window_size_new = window_size

                while True:
                    median_val = np.median(window)
                    max_val = np.max(window)
                    min_val = np.min(window)

                    if min_val < median_val < max_val:
                        if not min_val < input[i, j] < max_val:
                            output[i, j] = median_val
                        break
                    else:
                        window_size_new += 2
                        if window_size_new <= max_window_size:
                            pad_size_new = window_size_new // 2
                            window = input[max(0, i - pad_size_new) : min(input.shape[0], i + pad_size_new + 1),
                                           max(0, j - pad_size_new) : min(input.shape[1], j + pad_size_new + 1)]
                        else:
                            output[i, j] = median_val
                            break
        return output

    input_compensated = input.copy() * 255
    Fd = (input - scipy.ndimage.uniform_filter(input, 3, mode='mirror')) * 255.0

    for _i in range(num_iterations):
        halftone = dither_classic(input_compensated / 255, DITHER_KERNELS['floyd-steinberg'], False) * 255
        difference = skimage.filters.gaussian(np.abs((input * 255) - halftone), sigma=1.0)

        mean_difference = np.mean(difference)
        if mean_difference <= 0.0:
            continue

        u = uc / mean_difference
        height, width = input.shape
        for y in range(height):
            for x in range(width):
                C = 0.0
                if Fd[y, x] >= q:
                    C = u

                elif Fd[y, x] <= -q:
                    C = -u

                elif Fd[y, x] <= q:
                    C = (u / q**3.0) * (Fd[y, x]**3.0)

                input_compensated[y, x] = max(0, min(input_compensated[y, x] + C * difference[y, x], 255))

    if ramf_enabled:
        input_compensated = ramf(input_compensated, ramf_min, ramf_max)

    return dither_classic_modulated(input_compensated / 255, DITHER_KERNELS['floyd-steinberg'])


@click.group()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.option('--srgb-to-linear/--no-srgb-to-linear', help='Whether or not to convert the input image from sRGB to linear space before halftoning', default=False, show_default=True)
@click.option('--resize-width', type=int, help='Resizes the input image to the given width before halftoning')
@click.option('--resize-resample', type=click.Choice(['nearest', 'bilinear', 'bicubic', 'lanczos', 'box', 'hamming']), help='Resampling algorithm to use when resizing')
@click.option('--duplicate-first-rows-and-cols/--no-duplicate-first-rows-and-cols', default=False, show_default=True, help='Duplicates the first rows and columns of the image to avoid border startup-artefacts')
@click.option('--contrast', default=1.0, type=float, show_default=True, help='Adjusts contrast of the input image before halftoning, raise this value to increase contrast and vice versa')
@click.option('--sharpness', default=1.0, type=float, show_default=True, help='Adjusts sharpness of the input image before halftoning, raise this value to increase sharpness and vice versa')
@click.option('--brightness', default=1.0, type=float, show_default=True, help='Adjusts brightness of the input image before halftoning, raise this value to increase brightness and vice versa')
@click.option('--invert/--no-invert', default=False, show_default=True, help='Invert result *after* halftoning')
@click.option('--gamma', default=1.0, type=float, show_default=True, help='Manual gamma adjustment')
@click.option('--scale-low', default = 0.0, show_default=True, help='Performs scaling of the input image intensity levels by linear interpolation to this minimum value')
@click.option('--scale-high', default = 1.0, show_default=True, help='Performs scaling of the input image intensity levels by linear interpolation to this maximum value')
@click.pass_context
def cli(ctx, input, resize_width, resize_resample, duplicate_first_rows_and_cols, contrast, sharpness, brightness,
        gamma, srgb_to_linear, scale_low, scale_high, **kwargs):
    try:
        input_image = Image.open(input)
    except UnidentifiedImageError:
        raise click.BadParameter('Input file is not a valid image')

    if resize_width:
        resample_algorithm = None
        if resize_resample:
            resample_algorithm = getattr(Image.Resampling, resize_resample, None)

        ratio = resize_width / input_image.width
        input_image = input_image.resize((int(input_image.width * ratio), int(input_image.height * ratio)), resample=resample_algorithm)

    grayscale_image = input_image.convert('L')

    # Increasing Contrast/Sharpness might improve the legibility of smaller images
    # This will also reduce "fuzzyness" at the edges when dealing with logos and such
    if contrast != 1.0:
        enhance_contrast = ImageEnhance.Contrast(grayscale_image)
        grayscale_image = enhance_contrast.enhance(contrast)

    if sharpness != 1.0:
        enhance_sharpness = ImageEnhance.Sharpness(grayscale_image)
        grayscale_image = enhance_sharpness.enhance(sharpness)

    if brightness != 1.0:
        enhance_brightness = ImageEnhance.Brightness(grayscale_image)
        grayscale_image = enhance_brightness.enhance(brightness)

    # Duplicate first rows and columns of image to reduce artefacts
    # TODO: We might achieve the same effect by simply introducing a bit of noise in the first rows.
    if duplicate_first_rows_and_cols:
        kernel_size = 4

        canvas = Image.new("L", (grayscale_image.width + kernel_size, grayscale_image.height + kernel_size))

        # Fill first rows and first columns
        canvas.paste(grayscale_image, (kernel_size, 0))
        canvas.paste(grayscale_image, (0, kernel_size))
        canvas.paste(grayscale_image)
        canvas.paste(grayscale_image, (kernel_size, kernel_size))

        grayscale_image = canvas

    # Convert image to numpy array
    pixels = np.array(grayscale_image, dtype=np.float32,) / 255.0

    # Linear scaling to fit image intensities within target capabilities
    pixels = np.interp(pixels, (0.0, 1.0), (scale_low, scale_high))

    if gamma != 1.0:
        pixels = pixels ** (1.0 / gamma)

    if srgb_to_linear:
        pixels = np.where(pixels <= 0.04045, pixels/12.92, ((pixels+0.055)/1.055)**2.4)

    ctx.obj = pixels


@cli.result_callback()
@click.pass_obj
def process_result(input_obj, result, output, duplicate_first_rows_and_cols, invert, **kwargs):
    # TODO: Some of the papers contain details about how these metrics should be calculated. Make
    # sure that we're doing the exact same thing as those papers so that we can compare methods.
    #
    # It should be noted that metrics like these seems to be rather useless, as the simple
    # thresholding method produces images with the highest metrics, even though it obiously looks
    # the worst.
    #
    # And you can always improve the SSIM metric by sharpening the input image by ridiculous
    # amounts, easily "outperforming" all the other methods.
    #
    # The MSE/PSNR metrics are rather curious, as none of the methods seems to outperform plain old
    # Floyd-Steinberg. I'd love to see how TDED performs with this metric. It's easy to see
    # why it performs so good if you save the gaussian filtered images and compare them side by
    # side.
    psnr = skimage.metrics.peak_signal_noise_ratio(skimage.filters.gaussian(input_obj, sigma=1.0), skimage.filters.gaussian(result, sigma=1.0))
    mse = skimage.metrics.mean_squared_error(skimage.filters.gaussian(input_obj, sigma=1.0), skimage.filters.gaussian(result, sigma=1.0))
    ssim = skimage.metrics.structural_similarity(input_obj, result, win_size=11, data_range=1.0)
    ssim_wang = skimage.metrics.structural_similarity(input_obj, result, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

    click.echo("PSNR: {}, MSE: {}, SSIM: {}, SSIM-Wang: {}".format(psnr, mse, ssim, ssim_wang))

    result = np.uint8(np.rint(result * 255.0))
    image_output = Image.fromarray(np.uint8(result), "L")

    kernel_size = 4
    if duplicate_first_rows_and_cols:
        image_output = image_output.crop((kernel_size, kernel_size, image_output.width, image_output.height))

    if invert:
        image_output = ImageOps.invert(image_output)

    image_output.save(output)


@cli.command(help='Simple intensity thresholding, with optional noise applied to the threshold')
@click.option('--threshold', default=0.5)
@click.option('--noise', default=0.0)
@click.pass_obj
def threshold(input, threshold, noise):
    return dither_threshold(input, threshold, noise)


@cli.command(help='Random intensity thresholding')
@click.pass_obj
def whitenoise(input):
    return dither_whitenoise(input)


@cli.command(help='Blue noise thresholding')
@click.option('--noise-shape', type=(int, int), default=(64, 64))
@click.pass_obj
def bluenoise(input, noise_shape):
    return dither_bluenoise(input, noise_shape)


@cli.command(help='Ordered dithering by Bayer matrices')
@click.option('--matrix-size', default=16)
@click.pass_obj
def bayer(input, matrix_size):
    def dither_matrix(n):
        if n == 1:
            return np.array([[0]])
        else:
            first = (n ** 2) * dither_matrix(int(n/2))
            second = (n ** 2) * dither_matrix(int(n/2)) + 2
            third = (n ** 2) * dither_matrix(int(n/2)) + 3
            fourth = (n ** 2) * dither_matrix(int(n/2)) + 1
            first_col = np.concatenate((first, third), axis=0)
            second_col = np.concatenate((second, fourth), axis=0)
            return (1/n**2) * np.concatenate((first_col, second_col), axis=1)

    bayer_matrix = dither_matrix(matrix_size)

    return dither_bayer(input, bayer_matrix)


@cli.command(help='Classic 1970s error diffusion with variable kernels and optional serpentine path')
@click.option('--kernel', type=click.Choice(DITHER_KERNELS.keys()), default='floyd-steinberg')
@click.option('--serpentine/--no-serpentine', default=False)
@click.option('--k', default=0.0)
@click.option('--noise-multiplier', default=0.0)
@click.pass_obj
def classic(input, kernel, serpentine, k, noise_multiplier, **kwargs):
    return dither_classic(input, DITHER_KERNELS[kernel], serpentine, k, noise_multiplier)


@cli.command(help='A simple and efficient error diffusion algorithm')
@click.pass_obj
def ostromoukhov(input):
    return dither_ostromoukhov(input)


@cli.command()
@click.pass_obj
def zhoufang(input):
    def interpolate_and_mirror(key_levels):
        result = None
        key_prev, value_prev = key_levels[0]
        for key, value in key_levels[1:]:
            last_item = key == key_levels[-1][0]

            interpolation = np.linspace(value_prev, value, num=key - key_prev + (1 if last_item else 0), endpoint=last_item)

            if result is None:
                result = interpolation
            else:
                result = np.concatenate((result, interpolation))

            key_prev, value_prev = key, value

        result = np.concatenate((result, np.flip(result, axis=0)), axis=0)
        return result

    modulator_key_values = [
        (0, [0.0]),
        (44, [0.34]),
        (64, [0.50]),
        (85, [1.00]),
        (95, [0.17]),
        (102, [0.5]),
        (107, [0.7]),
        (112, [0.79]),
        (127, [1.00]),
    ]

    coefficients_key_values = [
        (0, [13, 0, 5]),
        (1, [1300249, 0, 499250]),
        (2, [213113, 287, 99357]),
        (3, [351854, 0, 199965]),
        (4, [801100, 0, 490999]),
        (10, [704075, 297466, 303694]),
        (22, [46613, 31917, 21469]),
        (32, [47482, 30617, 21900]),
        (44, [43024, 42131, 14826]),
        (64, [36411, 43219, 20369]),
        (72, [38477, 53843, 7678]),
        (77, [40503, 51547, 7948]),
        (85, [35865, 34108, 30026]),
        (95, [34117, 36899, 28983]),
        (102, [35464, 35049, 29485]),
        (107, [16477, 18810, 14712]),
        (112, [33360, 37954, 28685]),
        (127, [35269, 36066, 28664]),
    ]

    coefficients = interpolate_and_mirror(coefficients_key_values)
    sums = np.sum(coefficients, axis=1).reshape(256, 1)
    coefficients /= sums

    modulator = interpolate_and_mirror(modulator_key_values)

    return dither_zhoufang(input, coefficients, modulator)


@cli.command()
@click.option('--c', default=0.013, help='ITF coefficient, 0.013 is used by the paper, try 0.0065 for a "softer" look')
@click.pass_obj
def zhangpang(input, c):
    def interpolate_and_mirror(key_levels):
        result = None
        key_prev, value_prev = key_levels[0]
        for key, value in key_levels[1:]:
            last_item = key == key_levels[-1][0]

            interpolation = np.linspace(value_prev, value, num=key - key_prev + (1 if last_item else 0), endpoint=last_item)

            if result is None:
                result = interpolation
            else:
                result = np.concatenate((result, interpolation))

            key_prev, value_prev = key, value

        result = np.concatenate((result, np.flip(result, axis=0)), axis=0)
        return result

    modulator_key_values = [
        (0, [0.0]),
        (44, [0.34]),
        (64, [0.50]),
        (85, [1.00]),
        (95, [0.17]),
        (102, [0.5]),
        (107, [0.7]),
        (112, [0.79]),
        (127, [1.00]),
    ]

    coefficients_key_values = [
        (0, [13, 0, 5]),
        (1, [1300249, 0, 499250]),
        (2, [213113, 287, 99357]),
        (3, [351854, 0, 199965]),
        (4, [801100, 0, 490999]),
        (10, [704075, 297466, 303694]),
        (22, [46613, 31917, 21469]),
        (32, [47482, 30617, 21900]),
        (44, [43024, 42131, 14826]),
        (64, [36411, 43219, 20369]),
        (72, [38477, 53843, 7678]),
        (77, [40503, 51547, 7948]),
        (85, [35865, 34108, 30026]),
        (95, [34117, 36899, 28983]),
        (102, [35464, 35049, 29485]),
        (107, [16477, 18810, 14712]),
        (112, [33360, 37954, 28685]),
        (127, [35269, 36066, 28664]),
    ]

    coefficients = interpolate_and_mirror(coefficients_key_values)
    sums = np.sum(coefficients, axis=1).reshape(256, 1)
    coefficients /= sums

    modulator = interpolate_and_mirror(modulator_key_values)

    return dither_zhangpang(input, coefficients, modulator, c)


@cli.command()
@click.option('--kernel', type=click.Choice(DITHER_KERNELS.keys()), default='floyd-steinberg')
@click.option('--serpentine/--no-serpentine', default=False)
@click.option('-c', default=7.6, type=float, help='7.6 and 16.4 are mentioned in the paper')
@click.pass_obj
def entropy_constrained(input, kernel, serpentine, c, **kwargs):
    input_lowpass = skimage.filters.gaussian(input, sigma=1.0)
    input_highpass = input_lowpass - input

    return dither_entropy_constrained(input, input_highpass, c, DITHER_KERNELS[kernel], serpentine)


@cli.command()
@click.option('--kernel', type=click.Choice(DITHER_KERNELS.keys()), default='floyd-steinberg')
@click.option('-c', default=7.6, type=float, help='7.6 and 16.4 are mentioned in the paper')
@click.pass_obj
def entropy_constrained_ostromoukhov(input, kernel, c, **kwargs):
    input_lowpass = skimage.filters.gaussian(input, sigma=1.0)
    input_highpass = input_lowpass - input

    return dither_entropy_constrained_ostromoukhov(input, input_highpass, c)


@cli.command()
@click.option('--kernel', type=click.Choice(DITHER_KERNELS.keys()), default='floyd-steinberg')
@click.option('-c', default=7.6, type=float, help='7.6 and 16.4 are mentioned in the paper')
@click.pass_obj
def entropy_constrained_zhoufang(input, kernel, c, **kwargs):
    def interpolate_and_mirror(key_levels):
        result = None
        key_prev, value_prev = key_levels[0]
        for key, value in key_levels[1:]:
            last_item = key == key_levels[-1][0]

            interpolation = np.linspace(value_prev, value, num=key - key_prev + (1 if last_item else 0), endpoint=last_item)

            if result is None:
                result = interpolation
            else:
                result = np.concatenate((result, interpolation))

            key_prev, value_prev = key, value

        result = np.concatenate((result, np.flip(result, axis=0)), axis=0)
        return result

    modulator_key_values = [
        (0, [0.0]),
        (44, [0.34]),
        (64, [0.50]),
        (85, [1.00]),
        (95, [0.17]),
        (102, [0.5]),
        (107, [0.7]),
        (112, [0.79]),
        (127, [1.00]),
    ]

    coefficients_key_values = [
        (0, [13, 0, 5]),
        (1, [1300249, 0, 499250]),
        (2, [213113, 287, 99357]),
        (3, [351854, 0, 199965]),
        (4, [801100, 0, 490999]),
        (10, [704075, 297466, 303694]),
        (22, [46613, 31917, 21469]),
        (32, [47482, 30617, 21900]),
        (44, [43024, 42131, 14826]),
        (64, [36411, 43219, 20369]),
        (72, [38477, 53843, 7678]),
        (77, [40503, 51547, 7948]),
        (85, [35865, 34108, 30026]),
        (95, [34117, 36899, 28983]),
        (102, [35464, 35049, 29485]),
        (107, [16477, 18810, 14712]),
        (112, [33360, 37954, 28685]),
        (127, [35269, 36066, 28664]),
    ]

    coefficients = interpolate_and_mirror(coefficients_key_values)
    sums = np.sum(coefficients, axis=1).reshape(256, 1)
    coefficients /= sums

    modulator = interpolate_and_mirror(modulator_key_values)

    input_lowpass = skimage.filters.gaussian(input, sigma=1.0)
    input_highpass = input_lowpass - input

    return dither_entropy_constrained_zhoufang(input, coefficients, modulator, input_highpass, c)


@cli.command()
@click.option('--kernel', type=click.Choice(DITHER_KERNELS.keys()), default='floyd-steinberg')
@click.option('--serpentine/--no-serpentine', default=False)
@click.option('--scale-factor', type=float, default=0.5, help='0.85 yields sameish SSIM as paper, 0.5 is more discreet')
@click.pass_obj
def laplacian(input, kernel, serpentine, scale_factor):
    return dither_laplacian(input, DITHER_KERNELS[kernel], serpentine, scale_factor)


@cli.command()
@click.option('--mask-size', default=7)
@click.option('--k-parameter', default=2.6)
@click.pass_obj
def contrast_aware_basic(input, mask_size, k_parameter):
    return dither_contrast_aware_basic(input, mask_size, k_parameter)


@cli.command()
@click.option('--mask-size', default=7)
@click.option('--k-parameter', default=2.0)
@click.pass_obj
def contrast_aware_variant(input, mask_size, k_parameter):
    return dither_contrast_aware_variant(input, mask_size, k_parameter)


@cli.command()
@click.option('--num-iterations', default=3, help='3 and 5 is suggested in the paper')
@click.option('--uc', default=15.0, help='The uc parameter from the paper')
@click.option('--q', default=5.0, help='The q parameter from the paper')
@click.option('--ramf-enabled / --ramf-disabled', default=True, help='Disabling RAMF at smaller image sizes may yield better results')
@click.option('--ramf-min', default=3, help='Minimum size of the RAMF window')
@click.option('--ramf-max', default=11, help='Maximum size of the RAMF window')
@click.pass_obj
def visual_difference(input, num_iterations, uc, q, ramf_enabled, ramf_min, ramf_max):
    return dither_visual_difference(input, num_iterations, uc, q, ramf_enabled, ramf_min, ramf_max)


if __name__ == '__main__':
    cli()

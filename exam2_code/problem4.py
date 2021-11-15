#!/usr/bin/env python
import numpy as np
from PIL import Image

def stereo_match(left_img, right_img, kernel, max_offset):
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)
    w, h = left_img.size

    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w

    kernel_half = int(kernel / 2)
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range

    for y in range(kernel_half, h - kernel_half):
        print("\rProcessing.. %d%% complete"%(y / (h - kernel_half) * 100), end="", flush=True)

        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534

            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0

                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        ssd_temp = int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset])
                        ssd += ssd_temp * ssd_temp

                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset

            depth[y, x] = best_offset * offset_adjust

    Image.fromarray(depth).save('depth.png')

if __name__ == '__main__':
    stereo_match("exam2_pics/prob4_left_small.png", "exam2_pics/prob4_right_small.png", 6, 30)  # 6x6 local search kernel, 30 pixel search range
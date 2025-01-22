import numpy as np

def edgedetector(threshmap):
    rows, cols = threshmap.shape
    coastline = np.zeros((rows, cols), dtype=np.uint8)

    for xx in range(rows):
        for yy in range(cols):
            if threshmap[xx, yy] == 0:
                # Check bounds before accessing neighbors
                left = threshmap[xx, yy-1] if yy-1 >= 0 else 0
                right = threshmap[xx, yy+1] if yy+1 < cols else 0
                up = threshmap[xx-1, yy] if xx-1 >= 0 else 0
                down = threshmap[xx+1, yy] if xx+1 < rows else 0

                if xx == 0:  # Top row
                    if yy > 0 and (left == 0 and right == 0 and down == 0):
                        coastline[xx, yy] = 0
                    else:
                        coastline[xx, yy] = 255
                elif yy == 0:  # Left column
                    if xx > 0 and (up == 0 and right == 0 and down == 0):
                        coastline[xx, yy] = 0
                    else:
                        coastline[xx, yy] = 255
                else:  # General case
                    if (left == 0 and up == 0 and right == 0 and down == 0):
                        coastline[xx, yy] = 0
                    else:
                        coastline[xx, yy] = 255
            else:
                coastline[xx, yy] = 0

    return coastline

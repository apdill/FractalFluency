import numpy as np

def edgedetector(threshmap, max_iter):
    n = (2 ** max_iter) + 1
    coastline = np.zeros((n, n), dtype=np.uint8)

    for xx in range(n - 1):
        for yy in range(n - 1):
            if xx == 0 and yy > 0:
                if threshmap[xx, yy] == 0:
                    if (threshmap[xx, yy-1] == 0 and threshmap[xx, yy+1] == 0 and 
                        threshmap[xx+1, yy] == 0):
                        coastline[xx, yy] = 0
                    else:
                        coastline[xx, yy] = 255
                else:
                    coastline[xx, yy] = 0
            if yy == 0 and xx > 0:
                if threshmap[xx, yy] == 0:
                    if (threshmap[xx-1, yy] == 0 and threshmap[xx, yy+1] == 0 and 
                        threshmap[xx+1, yy] == 0):
                        coastline[xx, yy] = 0
                    else:
                        coastline[xx, yy] = 255
                else:
                    coastline[xx, yy] = 0
            if xx > 0 and yy > 0:
                if threshmap[xx, yy] == 0:
                    if (threshmap[xx, yy-1] == 0 and threshmap[xx-1, yy] == 0 and 
                        threshmap[xx, yy+1] == 0 and threshmap[xx+1, yy] == 0):
                        coastline[xx, yy] = 0
                    else:
                        coastline[xx, yy] = 255
                else:
                    coastline[xx, yy] = 0

    return coastline

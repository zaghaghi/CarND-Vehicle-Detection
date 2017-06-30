import numpy as np
import cv2

def find_cars(img, scale, ystart, ystop, cells_per_step, color):
    draw_img = np.copy(img)
    #img = img.astype(np.float32) / 255

    if ystart is None:
        ystart = 0
    if ystop is None:
        ystop = img.shape[0]
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = img_tosearch
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    nxblocks = (ctrans_tosearch.shape[1] // 16) - 2 + 1
    nyblocks = (ctrans_tosearch.shape[0] // 16) - 2 + 1

    window = 64
    nblocks_per_window = (window // 16) - 2 + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    bboxes = []
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            xleft = xpos* 16
            ytop = ypos* 16

            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            bboxes.append(((xbox_left, ytop_draw + ystart),
                            (xbox_left + win_draw, ytop_draw + win_draw+ystart)))
            cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                            (xbox_left + win_draw, ytop_draw+win_draw + ystart), color, 3)

    return bboxes, draw_img

scale_config1 = [(1.0, 400, 464, 1), (1.0, 416, 480, 1),
                (1.0, 432, 496, 1), (1.0, 448, 512, 1),
                (1.5, 400, 496, 1), (1.5, 432, 528, 1),
                (1.5, 464, 560, 1),
                (2.0, 400, 528, 1), (2.0, 432, 560, 1),
                (2.5, 400, 528, 1), (2.5, 432, 560, 1),
                (3.0, 400, 596, 1), (3.0, 464, 660, 1)]
scale_config2 = [(1.0, 400, 512, 1),
                (1.5, 400, 580, 1),
                (2.0, 400, 580, 1),
                (2.5, 420, 680, 1),
                (3.0, 380, 700, 1),
                (3.3, 380, 700, 1)]
#scale_config = [(3.31, 400, 720, 1), (2.0, 400, 720, 1), (1.5, 400, 550, 1), (1.0, 400, 500, 1)]
image1 = cv2.imread('video_images/54.png')
image2 = image1.copy()

colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255), (255,255,255),
          (128,0,0),(0,128,0),(0,0,128),(128,128,0),(128,0,128),(0,128,128), (128,128,128)]

bbs = 0
for i in range(len(scale_config1)):
    cfg = scale_config1[i]
    if cfg[0] == 3.3:
        bb, image1 = find_cars(image1, cfg[0], cfg[1], cfg[2], cfg[3], colors[i])
        bbs += len(bb)
        if len(bb) == 0:
            print(cfg)
        else:
            cv2.rectangle(image1, bb[0][0], bb[0][1], (0,0,0), 2)
print(bbs)
cv2.imwrite('bboxes1.png', image1)

bbs = 0
for i in range(len(scale_config2)):
    cfg = scale_config2[i]
    if cfg[0] == 3.3:
        bb, image2 = find_cars(image2, cfg[0], cfg[1], cfg[2], cfg[3], colors[i])
        bbs += len(bb)
        cv2.rectangle(image2, bb[0][0], bb[0][1], (0,0,0), 2)
print(bbs)
cv2.imwrite('bboxes2.png', image2)
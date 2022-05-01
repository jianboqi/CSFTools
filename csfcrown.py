#coding: utf-8
import gdal
import mahotas
import math
import numpy as np
import random
import sys
import argparse
import joblib
import tempfile
import os
from utils import saveToHdr
from utils import read_img_to_arr_no_transform

def log(*args):
    outstr = ""
    for i in args:
        outstr += str(i)
    print(outstr)
    sys.stdout.flush()


def sub_fun(_nuclear, _areas, _detected_trees_pos, _seg_index, _real_coordinate,
            _offset_x, _offset_y, _pixel_size, _seg_size):
    # 每棵树用不用的数值标记
    lower = _seg_index * _seg_size
    upper = min((_seg_index + 1) * _seg_size, _areas.max())
    log("Processing from: ", lower, " to ", upper)
    for i in range(lower+1, upper+1):
        treepixel = np.where(_areas == i)
        tmpmax = 0
        tx, ty = 0, 0 # row col
        tmp = 0
        crown = math.sqrt(len(treepixel[0]) * _pixel_size * _pixel_size/math.pi) * 2
        # tx,ty = sum(treepixel[0])/float(len(treepixel[0])),sum(treepixel[1])/float(len(treepixel[1]))
        for m in range(0, len(treepixel[0])):
            tmp += 1
            if _nuclear[treepixel[0][m]][treepixel[1][m]] > tmpmax:
                tmpmax = _nuclear[treepixel[0][m]][treepixel[1][m]]
                tx, ty = treepixel[0][m], treepixel[1][m]
        # maxHeight = nuclear[tx, ty]
        if _real_coordinate:
            x = _offset_x + ty * _pixel_size
            y = _offset_y + tx * _pixel_size
            _detected_trees_pos[i-1,:] = [x, y, crown]
        else:
            x = _offset_x + ty
            y = _offset_y + tx
            _detected_trees_pos[i-1, :] = [y, x, crown] # row col


def save_as_random_color_img(_dataarr, filepath):
    rows, cols = _dataarr.shape
    re = np.zeros((rows, cols, 3),dtype=np.uint8)
    colormap = dict()
    colormap[0.0] = [0, 0, 0]
    for row in range(rows):
        for col in range(cols):
            if _dataarr[row][col] in colormap:
                re[row, col, :] = colormap[_dataarr[row][col]]
            else:
                color = [random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)]
                re[row, col, :] = color
                colormap[_dataarr[row][col]] = color
    mahotas.imsave(filepath, re)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input CHM file.",required=True)
    parser.add_argument("-o", help="Output file.", required=True)
    parser.add_argument("-color_img", help="Save a color image.", type=bool, default=False)
    parser.add_argument("-seg_size", help="Tree number for each core. ", type=int, default=2000)
    parser.add_argument("-subregion", help="Divide image into subregions (pixels). ", type=int, default=1000)
    parser.add_argument("-height_threshold", help="Threshold to remove grass, bushes etc. ", type=float, default=2)
    parser.add_argument("-window_size", help="Window size for segmentation. ", type=int, default=7)
    parser.add_argument("-real_coordinate", help="Whether output real coordinates or pixel positions. ",
                        type=bool, default=True)
    args = parser.parse_args()

    import time
    start = time.clock()

    chm_hdr_path = args.i
    out_file = args.o
    subregion = args.subregion
    real_coordinate = args.real_coordinate
    seg_size = args.seg_size
    height_threshold = args.height_threshold
    window_size = args.window_size
    pixel_size = -1
    color_image = args.color_img

    idata_set = gdal.Open(chm_hdr_path)
    transform = idata_set.GetGeoTransform()
    if real_coordinate:
        if transform is None:
            log("ERROR: No geotransform found for file ", chm_hdr_path)
            sys.exit(0)
        else:
            pixel_size = abs(transform[1])


    band = idata_set.GetRasterBand(1)
    banddata = band.ReadAsArray(0, 0, band.XSize, band.YSize)
    width = band.XSize  # XSize是列数, YSize是行数
    height = band.YSize
    if (subregion == 0):  # 一般情况不会进行分块计算
        subregion = max(width, height)
    num_width = int(math.ceil(width / float(subregion)))
    num_height = int(math.ceil(height / float(subregion)))
    total_tree_num = 0
    log("INFO: Number of sub regions: " + str(num_height) + " * " + str(num_width))
    for num_r in range(0, num_height):  # row
        for num_c in range(0, num_width):  # col
            log("INFO: Region: " + str(num_r) + " " + str(num_c))
            row_start = num_r * subregion
            row_end = min((num_r + 1) * subregion, height)
            col_start = num_c * subregion
            col_end = min((num_c + 1) * subregion, width)
            if real_coordinate:
                offset_x = num_c * subregion * pixel_size
                offset_y = num_r * subregion * pixel_size
            else:
                offset_x = num_c * subregion
                offset_y = num_r * subregion

            nuclear = banddata[row_start:row_end, col_start:col_end]
            # 不保存文件 再读取，检测的树木数量很少，有问题以后检查
            row_col_img = chm_hdr_path + "_seg_"+str(num_r)+"_"+str(num_c)

            saveToHdr(nuclear,row_col_img)
            nuclear = read_img_to_arr_no_transform(row_col_img)
            if not color_image:
                os.remove(row_col_img)
                if os.path.exists(row_col_img + ".hdr"):
                    os.remove(row_col_img+".hdr")

            threshed = (nuclear > height_threshold)
            nuclear *= threshed
            bc = np.ones((window_size, window_size))

            maxima = mahotas.morph.regmax(nuclear, Bc=bc)
            spots, n_spots = mahotas.label(maxima)

            surface = (nuclear.max() - nuclear)
            areas = mahotas.cwatershed(surface, spots)
            areas *= threshed
            if color_image:
                save_as_random_color_img(areas, chm_hdr_path + "_seg_"+str(num_r)+"_"+str(num_c)+"color.jpg")

            area_max = areas.max()
            seg_num = int(math.ceil(area_max / float(seg_size)))
            total_tree_num += area_max
            log("INFO: Sub region trees:", area_max)
            log("INFO: Start parallel computing...")
            # temp file
            folder = tempfile.mkdtemp()
            detected_trees_pos = np.memmap(os.path.join(folder, 'treedetect'), dtype=float, shape=(area_max, 3),
                                           mode='w+')

            joblib.Parallel(n_jobs=joblib.cpu_count()-1, max_nbytes=None)(
                joblib.delayed(sub_fun)(nuclear, areas, detected_trees_pos, i,
                                        real_coordinate, offset_x, offset_y,
                                        pixel_size, seg_size) for i in range(0, seg_num))

            if num_r == 0 and num_c == 0:
                fw = open(out_file, 'w')
            else:
                fw = open(out_file, 'a')
            for i in range(0, len(detected_trees_pos)):
                outstr = str(detected_trees_pos[i][0]) + " " + str(detected_trees_pos[i][1])+\
                         " %.4f"%detected_trees_pos[i][2]
                fw.write(outstr + "\n")
            fw.close()
            del detected_trees_pos
            try:
                import shutil
                shutil.rmtree(folder)
            except OSError:
                log("Failed to delete: " + folder)
    log("INFO: Total detected trees: ", total_tree_num)
    log("Done.")
    end = time.clock()
    log("Time: ", "%.3fs" % (end - start))
import cv2
import json
import numpy as np
import textwrap

from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable


from sklearn import metrics

matplotlib.get_cachedir()
# plt.rc('font', family="Arial")

def add_value(S_set, json_file):
    single_mask = np.zeros_like(S_set[0])
    single_mask = single_mask.astype(np.float16)
    
    value_list_1 = np.array(json_file["smdl_score"])
    
    # value_list_2 = np.array(
    #     [1 - np.mean(json_file["org_score"]) + np.mean(json_file["baseline_score"])] + json_file["smdl_score"][:-1]
    # )
    value_list_2 = np.array(
        [np.mean(1 - np.array(json_file["insertion_score"][-1]) + np.array(json_file["deletion_score"][-1]))] + json_file["smdl_score"][:-1]
    )
    
    # value_list = np.exp((value_list_1 - value_list_2)/1)
    value_list = value_list_1 - value_list_2
    
    values = []
    value = 0
    i = 0
    for smdl_single_mask, smdl_value in zip(S_set, value_list):
        value = value - abs(smdl_value)
        single_mask[smdl_single_mask==1] = value
        values.append(value)
        i+=1
    attribution_map = single_mask - single_mask.min()
    attribution_map = attribution_map / attribution_map.max()
    
    return attribution_map, np.array(values)

def gen_cam(image_path, mask):
    """
    Generate heatmap
        :param image: [H,W,C]
        :param mask: [H,W],range 0-1
        :return: tuple(cam,heatmap)
    """
    # Read image
    w = mask.shape[1]
    h = mask.shape[0]
    image = cv2.resize(cv2.imread(image_path), (w,h))
    # mask->heatmap
    mask = cv2.resize(mask, (int(w/20),int(h/20)))
    mask = cv2.resize(mask, (w,h))
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_VIRIDIS)  # cv2.COLORMAP_COOL
    heatmap = np.float32(heatmap)

    # merge heatmap to original image
    cam = 0.5*heatmap + 0.5*np.float32(image)
    return cam.astype(np.uint8), (heatmap).astype(np.uint8)

def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def overlay_mask_with_white_edge(img, gt_mask, alpha=0.2, edge_thickness=1, edge_blur=0, rgb=True):
    """
    img:             可视化图 (H,W,3) 或 (H,W)，uint8，RGB(默认) 或 BGR（rgb=False）
    gt_mask:         二值/布尔 mask，(H,W)，>0 视为 True
    alpha:           填充透明度，0~1，越大越白
    edge_thickness:  白色边线粗细（像素）
    edge_blur:       边缘柔化半径（像素，偶数会自动+1），0 表示不柔化
    rgb:             True 表示 img 是 RGB；若你的图是 BGR（OpenCV常见），设为 False
    """
    # 统一成三通道 uint8
    if img.ndim == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if rgb else cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()
        if not rgb:
            # 内部统一按 RGB 计算，最后再换回
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    h, w = img_color.shape[:2]
    mask = (gt_mask > 0).astype(np.uint8)
    mask_3 = np.repeat(mask[:, :, None], 3, axis=2)

    # --- 1) 半透明白色覆盖（仅在 mask 内生效的局部 alpha 混合） ---
    img_float = img_color.astype(np.float32)
    white = np.full_like(img_float, 255, dtype=np.float32)
    # per-pixel alpha：mask 区域 alpha，其它地方 0
    a = (alpha * mask_3).astype(np.float32)
    blended = img_float * (1.0 - a) + white * a
    out = blended.astype(np.uint8)

    # --- 2) 取边界并画白线 ---
    # 方法A：形态学梯度，稳定且快
    k = 1 if edge_thickness <= 0 else edge_thickness
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)  # 0/1 边界

    # 可选：轻微柔化边缘遮罩，避免锯齿
    if edge_blur and edge_blur > 0:
        r = edge_blur + (edge_blur % 2 == 0)  # 保证为奇数
        edge = cv2.GaussianBlur(edge.astype(np.float32), (r, r), 0)
        edge = np.clip(edge, 0, 1)
    else:
        edge = edge.astype(np.float32)

    edge_3 = np.repeat(edge[:, :, None], 3, axis=2)
    # 白线直接叠加：把边界像素推向白色；edge 是 0~1，越接近 1 越白
    out = (out.astype(np.float32) * (1.0 - edge_3) + 255.0 * edge_3).astype(np.uint8)

    # 如原图是 BGR，需要转回以便直接用 cv2.imshow
    if not rgb:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out

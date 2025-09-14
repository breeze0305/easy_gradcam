import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

def save_heatmap(
        save_path,
        heat, 
        cmap="jet", 
        figsize=(4,4), 
        show_colorbar=False, 
        title=None
    ):
    plt.cla()
    plt.clf()
    plt.figure(figsize=figsize)
    sns.heatmap(
        heat,
        cmap=cmap,
        cbar=show_colorbar,
        xticklabels=False,
        yticklabels=False,
        square=True,
        annot=False
    )
    if title: 
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)



def save_mix_heatmap(
        save_path: str,
        heat: np.ndarray,
        ori_img: np.ndarray, # BGR 格式輸入
        cmap: str = "jet",
        alpha: float = 0.5, # 熱力圖透明度
        figsize: tuple = (4, 4),
        show_colorbar: bool = False,
    ):
    """
    繪製 heatmap 並與原圖疊加後存檔

    Args:
        save_path (str): 儲存路徑
        heat (np.ndarray): 2D heatmap array
        ori_img (np.ndarray): 原圖 (BGR 格式)
        cmap (str): 色彩映射
        alpha (float): 疊加透明度
        figsize (tuple): 圖片大小
        show_colorbar (bool): 是否顯示 colorbar
    """

    # 清空繪圖
    plt.cla()
    plt.clf()

    # 建立 heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        heat,
        cmap=cmap,
        cbar=show_colorbar,
        xticklabels=False,
        yticklabels=False,
        square=True,
        annot=False
    )

    plt.axis("off")
    plt.tight_layout()

    # 把 heatmap 輸出成 RGB 圖片
    plt.savefig("tmp_heatmap.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # 讀回熱力圖 (RGB)，再轉成 BGR
    heatmap = cv2.imread("tmp_heatmap.png")
    heatmap = cv2.resize(heatmap, (ori_img.shape[1], ori_img.shape[0]))
    os.remove("tmp_heatmap.png")

    # 疊加 heatmap 與原圖
    blended = cv2.addWeighted(ori_img, 1 - alpha, heatmap, alpha, 0)

    # 儲存結果
    cv2.imwrite(save_path, blended)
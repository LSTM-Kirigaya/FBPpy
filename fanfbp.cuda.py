from typing import Tuple
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
# import cupy as np
import torch

from PIL import Image
from joblib import Parallel, delayed

def getrphi(isect: np.ndarray, N: int, L: int) -> Tuple[np.ndarray, np.ndarray]:
    # 获取当前像素点的极坐标r phi  
    # 例如  
    cx = N / 2  
    cy = N / 2  
    i, j = np.unravel_index(isect, (N, N))  
    y = (N + 1 - i - cx - 0.5) * 2 * L / N  
    x = (j - cy - 0.5) * 2 * L / N  
    r = np.sqrt(x ** 2 + y ** 2)  
    phi = np.arctan(y / x)  
  
    # 归入0~2pi的范畴  
    phi[x < 0] += np.pi  
    phi[phi < 0] += 2 * np.pi  

    r = r.reshape(N, N)
    phi = phi.reshape(N, N)

    return r, phi
 

def fanfbp(pj: torch.Tensor, sp0: torch.Tensor, dp0: torch.Tensor, gridN: int, gridL: int, theta: torch.Tensor) -> torch.Tensor:  
  
    # 1.修正投影函数，注意将探测器平移成虚拟探测器  
    D = abs(sp0[0])  
    detector_num = len(dp0[0])  # 探测器个数
    angle_num = len(theta)  # 角度数  
    nT = dp0[1] * D / (D + dp0[0])  # 虚拟探测器的纵坐标  
    T = nT[1] - nT[0]  # 虚拟探测器间隔

    print(detector_num)

    pj = pj * D / torch.sqrt(D ** 2 + nT ** 2).cuda()
    pj = pj * (torch.abs(theta[1] - theta[0]))  

    # 2.卷积核（RL kernel h_RL和 SL Kernel h_SL）  
    # h_RL = ... (需要提供具体的卷积核函数)  
  
    r, phi = getrphi(np.arange(gridN ** 2), gridN, gridL)

    r = convert_tensor(r)
    phi = convert_tensor(phi)

    # 化为 三阶张量，一网打尽
    phi = phi.expand(angle_num, *phi.shape)
    r = r.expand(angle_num, *r.shape)
    beta = theta - torch.pi / 2
    beta = beta[:, None, None].expand(len(beta), gridN, gridN)

    th = torch.pi / 2 + beta + phi
    s = D * r * torch.sin(th) / (D + r * torch.cos(th))
    curdet = torch.floor((s - nT[0]) / T + 0.5).type(torch.int64)
    lam = (s - nT[0]) / T + 0.5 - curdet
    U = 1 + r * torch.sin(beta - phi) / D

    mask = (curdet > 0) & (curdet < detector_num - 1)
    curdet = curdet.clip(0, detector_num - 2)

    origin_shape = curdet.shape
    pj_curdet = pj.gather(1, curdet.reshape(angle_num, -1)).reshape(origin_shape)
    pj_curdet_1 = pj.gather(1, (curdet + 1).reshape(angle_num, -1)).reshape(origin_shape)

    I_s = torch.where(mask, ((1 - lam) * pj_curdet + lam * pj_curdet_1) / U ** 2, 0)

    return I_s.sum(axis=0)

def convert_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a).cuda()

if __name__ == '__main__':
    mat_data = h5py.File('./Kernel_Test.mat')
    pj = np.array(mat_data['pj']).T
    sp0 = np.array(mat_data['sp0']).T
    dp0 = np.array(mat_data['dp0']).T
    N = int(np.array(mat_data['grid']['N'])[0, 0])
    L = int(np.array(mat_data['grid']['L'])[0, 0])
    theta = np.array(mat_data['theta']).reshape(-1)

    pj = convert_tensor(pj)
    sp0 = convert_tensor(sp0)
    dp0 = convert_tensor(dp0)
    theta = convert_tensor(theta)

    s = time.time()
    with torch.no_grad():
        image = fanfbp(pj, sp0, dp0, N, L, theta)
    print('cost time', time.time() - s)
    

    image = (image - image.min()) / (image.max() - image.min())

    plt.imshow(image.detach().cpu(), cmap='gray')
    plt.axis('off')
    plt.show()
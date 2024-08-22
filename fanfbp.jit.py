from typing import Tuple
import time
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numba
from PIL import Image

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

@numba.njit()
def anti_prj_per_angle(gridN: int, beta, phi: np.ndarray, D, T, nT0, r, pj_t: np.ndarray):
    I = np.zeros((gridN, gridN))
    for i in range(gridN):  
        for j in range(gridN):  
            # 穿过该像素的射线与探测器的交点  
            th = np.pi / 2 + beta + phi[i, j]  # 角度的换算很麻烦 小心  
            s = D * r[i, j] * np.sin(th) / (D + r[i, j] * np.cos(th))  
            curdet = int((s - nT0) / T + 0.5)
            if curdet > 0 and curdet < N:  
                lam = (s - nT0) / T + 0.5 - curdet  
                U = 1 + r[i, j] * np.sin(beta - phi[i, j]) / D
                I[i, j] += ((1 - lam) * pj_t[curdet] + lam * pj_t[curdet + 1]) / U**2  

    return I

@numba.njit()
def make_RL_kernel(detector_num, T):
    kernel = np.zeros(detector_num)
    for i in range(detector_num):
        if i == detector_num // 2 - 1:
            kernel[i] = 1 / (4 * T ** 2)
        elif i % 2 == 1:
            kernel[i] = -1 / ((i - detector_num // 2 + 1) * np.pi * T) ** 2
    return kernel

@numba.njit()
def make_SL_kernel(detector_num, T):
    indices = np.arange(detector_num) - detector_num // 2 + 1
    kernel = - 2 / (np.pi * np.pi * T * T * (4 * indices * indices - 1))
    return kernel

def fanfbp(pj: np.ndarray, sp0: np.ndarray, dp0: np.ndarray, gridN: int, gridL: int, theta: np.ndarray, kernel_method: str) -> np.ndarray:
    # 1.修正投影函数，注意将探测器平移成虚拟探测器  
    D = abs(sp0[0])  
    detector_num = len(dp0[0])  # 探测器个数
    angle_num = len(theta)  # 角度数  
    nT = dp0[1] * D / (D + dp0[0])  # 虚拟探测器的纵坐标  
    T = nT[1] - nT[0]  # 虚拟探测器间隔  

    pj = pj * D / np.sqrt(D ** 2 + nT ** 2)
    pj = pj * (abs(theta[1] - theta[0]))  
  
    # 2.卷积核（RL kernel h_RL和 SL Kernel h_SL）
    if kernel_method == 'RL':
        kernel = make_RL_kernel(detector_num, T)
    elif kernel_method == 'SL':
        kernel = make_SL_kernel(detector_num, T)
    else:
        raise ValueError('kernel must be RL or SL')
    
    # 预先计算所有点的极坐标  
    r, phi = getrphi(np.arange(gridN ** 2), gridN, gridL)

    I = np.zeros((gridN, gridN))

    for t in range(angle_num):
        convres = np.convolve(pj[t], kernel, mode='full')
        # 要从 detector_num // 2 - 1 处开始取值，因为构造的滤波器的 detector_num // 2 - 1 刚好是中心，两边对称
        pj_t = convres[detector_num // 2 - 1: detector_num // 2 - 1 + len(pj[t])]
        I += anti_prj_per_angle(gridN, theta[t] - np.pi / 2, phi, D[0], T, nT[0], r, pj_t)

    return I


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='./Kernel_Test.mat')
    parser.add_argument('-k', default='RL', choices=['RL', 'SL'])

    args = parser.parse_args()

    mat_data = h5py.File(args.i)

    pj = np.array(mat_data['pj']).T
    sp0 = np.array(mat_data['sp0']).T
    dp0 = np.array(mat_data['dp0']).T
    N = int(np.array(mat_data['grid']['N'])[0, 0])
    L = int(np.array(mat_data['grid']['L'])[0, 0])
    theta = np.array(mat_data['theta']).reshape(-1)

    s = time.time()
    image = fanfbp(pj, sp0, dp0, N, L, theta, args.k)    
    image = (image - image.min()) / (image.max() - image.min())

    if image.max() > 0:
        image = image * 255.
    
    image = image.astype(np.uint8)

    save_path = '{}_{}.png'.format(args.i, args.k)
    Image.fromarray(image).save(save_path)

    print('cost time {} s'.format(round(time.time() - s, 2)), ', image has been saved to ' + save_path)
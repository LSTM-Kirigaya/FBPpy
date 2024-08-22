from typing import Tuple
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

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

def anti_prj_per_angle(detector_num: int, beta, phi: np.ndarray, D, T, nT0, r, pj_t: np.ndarray):
    th = np.pi / 2 + beta + phi
    s = D * r * np.sin(th) / (D + r * np.cos(th))
    curdet = np.int64((s - nT0) / T + 0.5)
    mask = (curdet > 0) & (curdet < detector_num - 1)
    lam = (s - nT0) / T + 0.5 - curdet
    U = 1 + r * np.sin(beta - phi) / D

    curdet[(curdet + 1) >= pj_t.size] = 0
    pj_t_1 = np.take(pj_t, curdet)
    pj_t_2 = np.take(pj_t, curdet + 1)
    
    I = np.where(mask, ((1 - lam) * pj_t_1 + lam * pj_t_2) / U ** 2, 0)

    return I

def fanfbp(pj: np.ndarray, sp0: np.ndarray, dp0: np.ndarray, gridN: int, gridL: int, theta: np.ndarray) -> np.ndarray:  
  
    # 1.修正投影函数，注意将探测器平移成虚拟探测器
    D = abs(sp0[0])  
    detector_num = len(dp0[0])  # 探测器个数
    angle_num = len(theta)  # 角度数  
    nT = dp0[1] * D / (D + dp0[0])  # 虚拟探测器的纵坐标  
    T = nT[1] - nT[0]  # 虚拟探测器间隔  

    pj = pj * D / np.sqrt(D ** 2 + nT ** 2)
    pj = pj * (abs(theta[1] - theta[0]))  
  
    # 2.卷积核（RL kernel h_RL和 SL Kernel h_SL）  
    # h_RL = ... (需要提供具体的卷积核函数)  
  
    # 预先计算所有点的极坐标  
    r, phi = getrphi(np.arange(gridN ** 2), gridN, gridL)  
    
    # 3.卷积加权反投影
    parallel = Parallel(n_jobs=-1)
    results = parallel(
        delayed(anti_prj_per_angle)(detector_num, theta[t] - np.pi / 2, phi, D[0], T, nT[0], r, pj[t])
        for t in range(angle_num)
    )

    return sum(results)


if __name__ == '__main__':
    mat_data = h5py.File('./Kernel_Test.mat')
    pj = np.array(mat_data['pj']).T
    sp0 = np.array(mat_data['sp0']).T
    dp0 = np.array(mat_data['dp0']).T
    N = int(np.array(mat_data['grid']['N'])[0, 0])
    L = int(np.array(mat_data['grid']['L'])[0, 0])
    theta = np.array(mat_data['theta']).reshape(-1)

    s = time.time()
    image = fanfbp(pj, sp0, dp0, N, L, theta)
    print('cost time', time.time() - s)
    

    image = (image - image.min()) / (image.max() - image.min())

    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    if image.max() > 0:
        image = image * 255.
    
    image = image.astype(np.uint8)
    Image.fromarray(image).save('test.png')
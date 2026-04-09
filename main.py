import numpy as np
import cv2

def warp_conformal(image_path, output_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    x = np.linspace(-2, 2, w)
    y = np.linspace(-2, 2, h)
    xv, yv = np.meshgrid(x, y)

    z = xv + 1j * yv

    s = 2
    cT = 1 - (np.log(s)**2)/(4*np.pi**2)+(np.log(s)/(2*np.pi))*1j
    z0 = 0 + 0j
    A = np.exp(z0*(1-cT))

    with np.errstate(divide='ignore', invalid='ignore'):
        f_z = np.pow(z,cT)*A

    u = np.real(f_z)
    v = np.imag(f_z)

    map_x = ((u + 4) / 8 * (w - 1)).astype(np.float32)
    map_y = ((v + 4) / 8 * (h - 1)).astype(np.float32)

    warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imwrite(output_path, warped_img)
    print(f"Warped image saved to {output_path}")

warp_conformal('self_similar.png', 'escher_image.jpg')

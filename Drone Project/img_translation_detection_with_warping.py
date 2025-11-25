"""
detect warp → pick best method → return Δx, Δy

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# pre-process: blur → edges → Hann
def pre_process(gray, sigma=1.0, edges=True):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    if edges:
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
        _, bw = cv2.threshold(mag, 0, 255, cv2.THRESH_OTSU)
        img = bw.astype(np.float32) / 255
    else:
        img = blur.astype(np.float32) / 255
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    return img * hann


# estimate Homography  (ORB + BFMatcher + RANSAC)
def estimate_H(img1, img2, n_features=6000):
    orb = cv2.ORB_create(n_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, None         

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt  for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
    return H, mask



# (rotation≈0, scale≈1, shear≈0, no perspective)
# def looks_like_translation(H, tol_rot=1.5, tol_scale=0.02,
#                            tol_shear=0.015, tol_persp=1e-4):
#     if H is None:
#         return False
#     # normalise so H[2,2] == 1
#     H = H / H[2, 2]
#     a, b, tx = H[0]
#     c, d, ty = H[1]
#     g, h, _  = H[2]

#     # perspective elements small?
#     if abs(g) > tol_persp or abs(h) > tol_persp:
#         return False
#     # rotation matrix part
#     rot = np.rad2deg(np.arctan2(-b, a))
#     if abs(rot) > tol_rot:
#         return False
#     # scale close to 1
#     scale_x, scale_y = np.hypot(a, b), np.hypot(c, d)
#     if abs(scale_x - 1) > tol_scale or abs(scale_y - 1) > tol_scale:
#         return False
#     # shear small: dot product between rows ≈ 0
#     shear = abs(a*c + b*d) / (scale_x * scale_y)
#     if shear > tol_shear:
#         return False
#     return True



def find_dx_dy(reference_path, live_path, return_warped=False):
    ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    live = cv2.imread(live_path, cv2.IMREAD_GRAYSCALE)
    if ref is None or live is None:
        raise IOError("image paths invalid")

    if live.shape != ref.shape:
        live = cv2.resize(live, (ref.shape[1], ref.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    # Estimate homography 
    H, _ = estimate_H(ref, live)

    if looks_like_translation(H):
        # Un-warped case → direct phase correlate 
        ref_p  = pre_process(ref)
        live_p = pre_process(live)
        (dx, dy), snr = cv2.phaseCorrelate(ref_p, live_p)
        warped = live                    
    else:
        # Warped case -> rectify -> phase correlate
        live_rect = cv2.warpPerspective(live, H, (ref.shape[1], ref.shape[0]))
        ref_p  = pre_process(ref)
        rect_p = pre_process(live_rect)
        (dx, dy), snr = cv2.phaseCorrelate(ref_p, rect_p)
        warped = live_rect

    result = {"dx": dx, "dy": dy, "snr": snr}
    if return_warped:
        result["warped"] = warped
        result["H"] = H
    return result



if __name__ == "__main__":
    ref_img  = "reference.jpeg"
    live_img = "live.jpeg"

    out = find_dx_dy(ref_img, live_img, return_warped=True)

    print(f"\nΔx = {out['dx']:.3f}   Δy = {out['dy']:.3f}   "
          f"SNR = {out['snr']:.3f}")
    
    ref = cv2.imread(ref_img,  cv2.IMREAD_GRAYSCALE)
    warped = out["warped"]
    overlay = cv2.addWeighted(ref, 0.5, warped, 0.5, 0)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(ref, cmap='gray');    plt.title("Reference");  plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(warped, cmap='gray'); plt.title("Live (rectified)"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(overlay, cmap='gray');plt.title("Overlay");    plt.axis('off')
    
    plt.suptitle(f"Δx={out['dx']:.2f}  Δy={out['dy']:.2f}", fontsize=14)
    plt.tight_layout(); plt.show()

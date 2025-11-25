import cv2
import numpy as np
import matplotlib.pyplot as plt


REF_IMG  = "img1.jpeg"   # reference
LIVE_IMG = "img2.jpeg"   # live / shifted


ref_gray  = cv2.imread(REF_IMG,  cv2.IMREAD_GRAYSCALE)
live_gray = cv2.imread(LIVE_IMG, cv2.IMREAD_GRAYSCALE)
if ref_gray is None or live_gray is None:
    raise IOError("Check image file names.")

h, w = ref_gray.shape
if live_gray.shape != (h, w):
    live_gray = cv2.resize(live_gray, (w, h), interpolation=cv2.INTER_LINEAR)


def preprocess(gray, sigma=1.0, use_edges=True):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    if use_edges:
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.convertScaleAbs(mag)
        _, binmap = cv2.threshold(mag, 0, 255, cv2.THRESH_OTSU)
        proc = binmap.astype(np.float32) / 255.0
    else:
        proc = blur.astype(np.float32) / 255.0
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    return proc * hann

ref_pre  = preprocess(ref_gray)
live_pre = preprocess(live_gray)


F1 = np.fft.fft2(ref_pre)
F2 = np.fft.fft2(live_pre)
R  = F1 * np.conj(F2)
R /= np.maximum(np.abs(R), 1e-10)     
corr = np.fft.ifft2(R)
corr = np.fft.fftshift(corr.real)          # move (0,0) to centre

# find integer‑pixel peak
peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
cy, cx = h // 2, w // 2
dx_int = peak_x - cx
dy_int = peak_y - cy

# 3×3 quadratic sub‑pixel fit
def subpixel(c):
    y, x = c
    if 0 < y < h-1 and 0 < x < w-1:
        win = corr[y-1:y+2, x-1:x+2]
        dx = (win[1,2] - win[1,0]) / (2*(2*win[1,1]-win[1,0]-win[1,2]))
        dy = (win[2,1] - win[0,1]) / (2*(2*win[1,1]-win[0,1]-win[2,1]))
        return dx, dy
    return 0.0, 0.0

sub_dx, sub_dy = subpixel((peak_y, peak_x))
dx = dx_int + sub_dx
dy = dy_int + sub_dy


snr = corr[peak_y, peak_x] / (np.mean(corr) + 1e-9)

fig, ax = plt.subplots(figsize=(7,7))
ax.imshow(ref_gray, cmap='gray')
ax.set_title("Reference with translation arrow")
ax.set_xlim(0, w); ax.set_ylim(h, 0)
ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
ax.grid(color='white', alpha=0.3)

cx, cy = cx, cy
ax.plot(cx, cy, 'bo', label="Reference centre")
ax.plot(cx + dx, cy + dy, 'ro', label="Live centre")

ax.arrow(cx, cy, dx, dy, color='cyan', width=0.5, head_length=10,
         head_width=6, length_includes_head=True)
ax.text(cx + dx/2, cy + dy/2,
        f"Δx={dx:.2f}\nΔy={dy:.2f}", color='yellow',
        ha='center', va='center',
        bbox=dict(facecolor='black', alpha=0.4, pad=3))
ax.legend(facecolor='white')
plt.tight_layout()
plt.show()

print(f"Measured shift Δx={dx:.2f}px, Δy={dy:.2f}px  (integer: {dx_int},{dy_int})  SNR≈{snr:.2f}")

import os
import re
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

SEED = 42
random.seed(SEED)
rng = np.random.default_rng(seed=SEED)

# Remove all existing images in the directory that start with 'HW1_IMG_CS898BA' and end with '.png'
# Except for the original image
for f in os.listdir('.'):
    if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png') and f != 'HW1_IMG_CS898BA.png':
        os.remove(f)

image = cv2.imread('HW1_IMG_CS898BA.png')
print(image.shape)

# Part 2:
# 1. Find and print basic image statistics of the original image for each individual channel (min, max, average, median, mode, skew, range, standard deviation, variance)
blue, green, red = cv2.split(image)

print("Blue channel:")
print("    min:", blue.min())
print("    max:", blue.max())
print("    average:", blue.mean())
print("    median:", np.median(blue))
print("    mode:", np.bincount(blue.flatten()).argmax())
print("    skew:", (blue.mean() - np.median(blue)) / blue.std())
print("    range:", blue.max() - blue.min())
print("    std:", blue.std())
print("    variance:", blue.var())

print("Green channel:")
print("    min:", green.min())
print("    max:", green.max())
print("    average:", green.mean())
print("    median:", np.median(green))
print("    mode:", np.bincount(green.flatten()).argmax())
print("    skew:", (green.mean() - np.median(green)) / green.std())
print("    range:", green.max() - green.min())
print("    std:", green.std())
print("    variance:", green.var())

print("Red channel:")
print("    min:", red.min())
print("    max:", red.max())
print("    average:", red.mean())
print("    median:", np.median(red))
print("    mode:", np.bincount(red.flatten()).argmax())
print("    skew:", (red.mean() - np.median(red)) / red.std())
print("    range:", red.max() - red.min())
print("    std:", red.std())
print("    variance:", red.var())

# 2. Convert and save the image to greyscale, binary, and different color spaces (HSV, CIELAB, and HLS).
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('HW1_IMG_CS898BA_grayscale.png', grayscale_image)

# Binary image using Adaptive Gaussian Thresholding
binary_image = cv2.adaptiveThreshold(
    src=grayscale_image,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)
cv2.imwrite('HW1_IMG_CS898BA_binary.png', binary_image)

HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite('HW1_IMG_CS898BA_HSV.png', HSV_image)

CIELAB_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imwrite('HW1_IMG_CS898BA_CIELAB.png', CIELAB_image)

HLS_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
cv2.imwrite('HW1_IMG_CS898BA_HLS.png', HLS_image)

# 3. On the HSV converted image, normalize the lighting by performing histogram equalization across the V (value) channel.
hue, saturation, value = cv2.split(HSV_image)
equalized_value = cv2.equalizeHist(value)
equalized_HSV = cv2.merge((hue, saturation, equalized_value))

# 4. Convert the normalized image back to RGB and save it.
equalized_image = cv2.cvtColor(equalized_HSV, cv2.COLOR_HSV2BGR)
cv2.imwrite('HW1_IMG_CS898BA_equalized.png', equalized_image)

# 5. You should now have 7 images.
print("Total images in directory:", len([f for f in os.listdir('.') if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png')]))

# 6. Perform random affine transformations on each image (you should perform 14 total transformations - 2 for each image). Affine transformations can be translation, rotation, scaling, or shear as long as each is unique in either transformation type or transformation value (rotate 90 degrees vs rotate 186 degrees). No two images should be transformed in the exact same way. Save each of those images to new files.

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height))

def shear_image(image, shear_matrix):
    height, width = image.shape[:2]
    pts1 = np.float32([[0, 0], [width, 0], [0, height]])
    pts2 = np.float32(shear_matrix) * np.float32([width, height])
    matrix = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, matrix, (width, height))

def scale_image(image, scale_factor):
    height, width = image.shape[:2]
    cx, cy = width / 2, height / 2
    matrix = np.float32([
        [scale_factor, 0, cx * (1 - scale_factor)],
        [0, scale_factor, cy * (1 - scale_factor)],
    ])
    interp = cv2.INTER_CUBIC if scale_factor >= 1.0 else cv2.INTER_AREA
    return cv2.warpAffine(image, matrix, (width, height), flags=interp)

def translate_image(image, translation):
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    return cv2.warpAffine(image, matrix, (width, height))

transformations_dict = {
    '': [ # Original image
        [('rotate', 20.2), ('scale', 1.3), ('translate', (50, 30))],
        [('rotate', -15.6), ('shear', [[0, 0], [1, 0], [0.15, 1]])]
    ],
    'grayscale': [
        [('scale', 1.45), ('rotate', -22.3), ('translate', (15, -10)), ('shear', [[0, 0], [1, 0], [0.08, 1]])],
        [('translate', (25, -18)), ('rotate', 11.7)]
    ],
    'binary': [
        [('translate', (45, 35)), ('scale', 0.82)],
        [('rotate', 47.5), ('shear', [[0, 0], [1, -0.12], [0, 1]]), ('scale', 1.22), ('translate', (-20, 18))]
    ],
    'HSV': [
        [('shear', [[0, 0], [1, -0.27], [0, 1]]), ('translate', (-40, 22)), ('rotate', 8.9)],
        [('rotate', 63.4), ('scale', 1.18)]
    ],
    'CIELAB': [
        [('rotate', 90.0), ('translate', (32, -52)), ('scale', 1.05), ('shear', [[0, 0], [1, 0.06], [0, 1]])],
        [('scale', 0.73), ('shear', [[0, 0], [1, 0.21], [0, 1]])]
    ],
    'HLS': [
        [('scale', 0.78), ('shear', [[0, 0], [1, 0], [0.18, 1]])],
        [('rotate', -38.7), ('translate', (-28, 41)), ('scale', 0.91), ('shear', [[0, 0], [1, -0.05], [0.03, 1]])]
    ],
    'equalized': [
        [('rotate', 74.9), ('scale', 1.07), ('translate', (8, 14)), ('shear', [[0, 0], [1, 0.09], [0, 1]]), ('rotate', -5.2)],
        [('translate', (12, -58)), ('shear', [[0, 0], [1, 0], [0.24, 1]])]
    ],
}

for f in os.listdir('.'):
    if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png'):
        image_type = f.split('.')[0].split('_')[-1]
        if image_type == 'CS898BA':  # Original image
            image_type = ''
        if image_type in transformations_dict:
            img = cv2.imread(f)
            for i, transformations in enumerate(transformations_dict[image_type]):
                transformed_img = img.copy()
                for (transformation, value) in transformations:
                    if transformation == 'rotate':
                        transformed_img = rotate_image(transformed_img, value)
                    elif transformation == 'shear':
                        transformed_img = shear_image(transformed_img, value)
                    elif transformation == 'scale':
                        transformed_img = scale_image(transformed_img, value)
                    elif transformation == 'translate':
                        transformed_img = translate_image(transformed_img, value)
                new_filename = f.replace('.png', f'_transformed_{i+1}.png')
                cv2.imwrite(new_filename, transformed_img)

# 7. You should now have 21 images.
print("Total images in directory after transformations:", len([f for f in os.listdir('.') if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png')]))

# 8. Apply a Gaussian blur to each image using the levels of sigma: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5. Discuss how the level of sigma changes the image. Save each of those images to new files.
sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
for f in os.listdir('.'):
    if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png'):
        img = cv2.imread(f)
        for sigma in sigma_values:
            blurred_img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
            new_filename = f.replace('.png', f'_blurred_sigma{sigma}.png')
            cv2.imwrite(new_filename, blurred_img)

# Low sigma values like 0.5 and 1.0 helps in smoothing the images while preserving most of the details.
# As the sigma value increases, the images lose more and more details like edges and textures, which makes it harder to detect edges later on.

# 9. You should now have 168 images.
print("Total images in directory after blurring:", len([f for f in os.listdir('.') if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png')]))

# Part 3:
# 1. Randomly create 4 equally sized subsets of the images from part 2.
all_images = [f for f in os.listdir('.') if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png')]
rng.shuffle(all_images)
subset_size = len(all_images) // 4
subsets = [all_images[i:i + subset_size] for i in range(0, len(all_images), subset_size)]
for i, subset in enumerate(subsets):
    print(f"Subset {i+1}: {len(subset)} images")

# 2. Choose a subset to use in the remaining steps.
chosen_subset = subsets[0]

# 3. You should now have 42 images.
print("Total images in chosen subset:", len(chosen_subset))

# 4. Perform these edge detection techniques on that subset:
# Remove all other images in the directory that are not in the chosen subset
for f in os.listdir('.'):
    if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png') and f not in chosen_subset and f != 'HW1_IMG_CS898BA.png':
        os.remove(f)

for f in chosen_subset:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # a.  Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.convertScaleAbs(sobel_magnitude)
    cv2.imwrite(f.replace('.png', '_sobel.png'), sobel)

    # b.  Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    cv2.imwrite(f.replace('.png', '_laplacian.png'), laplacian)

    # c.  Canny
    median = float(np.median(gray))
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    canny = cv2.Canny(gray, lower, upper)
    cv2.imwrite(f.replace('.png', '_canny.png'), canny)

    # d.  Prewitt
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
    prewitt_magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    prewitt = cv2.convertScaleAbs(prewitt_magnitude)
    cv2.imwrite(f.replace('.png', '_prewitt.png'), prewitt)

# 5. Discuss the pros and cons of each edge detection technique and perform an analysis of which of these techniques works best for this image set.
# Sobel:
# Pros - Details edges in both horizontal and vertical directions.
# Cons - May produce thick and/or non-connected edges, sensitive to noise.
# Laplacian:
# Pros - Detects edges in all directions, good for finding fine details.
# Cons - Very sensitive to noise, may produce false edges.
# Canny:
# Pros - Good for detecting edges in noisy images, provides thin and connected edges, uses non-maximum suppression.
# Cons - Requires tuning of threshold cutoffs, may miss weak edges or produce false edges.
# Prewitt:
# Pros - Simple and fast, good for detecting edges in clean images.
# Cons - May produce thicker edges, sensitive to noise.

# For this image set, Sobel is the best because it provides clear edges out of the most images, while Laplacian and Prewitt produce more noise and Canny misses some edges in the blurred images. However, for bright images, Prewitt can perform better than Sobel because Sobel produce too many edges.

# 6. Save each image before and after adding edges with each technique.
# Already done in the code in Part 4.

# 7. You should now have 210 images.
# Check if the original image is in chosen subset, if not, ignore it in the count since it does not have edge detection applied to it.
if 'HW1_IMG_CS898BA.png' in chosen_subset:
    print("Total images in directory after edge detection:", len([f for f in os.listdir('.') if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png')]))
else:
    print("Total images in directory after edge detection (excluding original):", len([f for f in os.listdir('.') if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png')]) - 1)

# 8. Create 42, 5-image plots of the input image (from the start of part 3) next to the edge-detected images and output 6 random plots to add to the readme. Include information on what processing techniques were used on the images.
# Text at the top:
# Original
# → BGR (Original), Grayscale, Binary, HSV, CIELAB, HLS, Equalized
# → Affine(Rot:value°, Scale:value, Translate:[x,y], Shear:[Sx:value,Sy:value])
# → Gaussian Blur(σ:value)
# 5 images layout (3x3):
# _, Sobel Edge, _
# Laplacian Edge, Input Image, Canny Edge
# _, Prewitt Edge, _
COLOR_SPACE_LABEL = {
    '': 'BGR',
    'grayscale': 'Grayscale',
    'binary': 'Binary',
    'HSV': 'HSV',
    'CIELAB': 'CIELAB',
    'HLS': 'HLS',
    'equalized': 'Equalized',
}
BG_COLOR = '#363636'
TEXT_COLOR = 'white'

def parse_filename(fname):
    rest = fname.replace('.png', '')[len('HW1_IMG_CS898BA'):].lstrip('_')
    sigma_match = re.search(r'blurred_sigma([\d.]+)', rest)
    sigma = float(sigma_match.group(1)) if sigma_match else None
    rest = re.sub(r'_?blurred_sigma[\d.]+', '', rest)
    trans_match = re.search(r'transformed_(\d+)', rest)
    trans_idx = int(trans_match.group(1)) - 1 if trans_match else None
    rest = re.sub(r'_?transformed_\d+', '', rest)
    return rest.strip('_'), trans_idx, sigma

def get_transformations(transforms):
    parts = []
    for name, val in transforms:
        if name == 'rotate':
            parts.append(f'Rot:{val}°')
        elif name == 'scale':
            parts.append(f'Scale:{val}')
        elif name == 'translate':
            parts.append(f'Trans:[{val[0]},{val[1]}]')
        elif name == 'shear':
            parts.append(f'Shear:[Sx:{val[1][1]},Sy:{val[2][0]}]')
    return 'Affine(' + ', '.join(parts) + ')'

def build_title(fname):
    cs_key, trans_idx, sigma = parse_filename(fname)
    lines = ['Original', f'→ {COLOR_SPACE_LABEL[cs_key]}']
    if trans_idx is not None:
        lines.append(f'→ {get_transformations(transformations_dict[cs_key][trans_idx])}')
    if sigma is not None:
        lines.append(f'→ Gaussian Blur(σ:{sigma})')
    return '\n'.join(lines)

def to_rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None and im.ndim == 3 else im

os.makedirs('plots', exist_ok=True)
for old_image in os.listdir('plots'):
    if old_image.endswith('.png'):
        os.remove(os.path.join('plots', old_image))
plot_paths = []
for i, f in enumerate(chosen_subset):
    img = cv2.imread(f)
    sobel = cv2.imread(f.replace('.png', '_sobel.png'))
    laplacian = cv2.imread(f.replace('.png', '_laplacian.png'))
    canny = cv2.imread(f.replace('.png', '_canny.png'))
    prewitt = cv2.imread(f.replace('.png', '_prewitt.png'))

    fig = plt.figure(figsize=(10, 12), facecolor=BG_COLOR)
    fig.text(0.5, 0.97, build_title(f), ha='center', va='top',
             color=TEXT_COLOR, fontsize=12, family='monospace', linespacing=1.5)

    gs = GridSpec(3, 3, figure=fig, top=0.76, bottom=0.04,
                  left=0.04, right=0.96, hspace=0.25, wspace=0.1)
    panels = {
        (0, 1): ('Sobel Edge', sobel),
        (1, 0): ('Laplacian Edge', laplacian),
        (1, 1): ('Input Image', img),
        (1, 2): ('Canny Edge', canny),
        (2, 1): ('Prewitt Edge', prewitt),
    }
    for (r, c), (title, im) in panels.items():
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(to_rgb(im))
        ax.set_title(title, color=TEXT_COLOR, fontsize=11)
        ax.set_facecolor(BG_COLOR)
        ax.axis('off')

    out_path = os.path.join('plots', f.replace('.png', '_plot.png'))
    fig.savefig(out_path, facecolor=BG_COLOR, dpi=100)
    plt.close(fig)
    plot_paths.append(out_path)

readme_samples = random.sample(plot_paths, min(6, len(plot_paths)))
print('Plots selected for README:')
for plot in readme_samples:
    print(' ', plot)

# Update the README to display the selected plots under "# HW 1 Output Examples",
# preserving any sections (e.g. "# Discussions") that follow.
README_PATH = 'README.md'
README_MARKER = '# HW 1 Output Examples'
if os.path.exists(README_PATH):
    with open(README_PATH, 'r', encoding='utf-8') as f:
        readme_content = f.read()
    if README_MARKER in readme_content:
        head, _, tail = readme_content.partition(README_MARKER)
        next_section = re.search(r'^---\s*$', tail[1:], re.MULTILINE)
        rest = tail[1 + next_section.start():] if next_section else ''
        image_md = '\n\n' + '\n\n'.join(
            f'![{os.path.basename(plot)}]({plot.replace(os.sep, "/")})'
            for plot in readme_samples
        ) + '\n\n'
        new_content = head + README_MARKER + image_md + rest
        with open(README_PATH, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Updated {README_PATH} with {len(readme_samples)} plot images.')
    else:
        print(f'Marker "{README_MARKER}" not found in {README_PATH}; README not updated.')
else:
    print(f'{README_PATH} not found; skipping README update.')

# System requirement
- Python 3.11

---

# Code Explanations


---

# Setting up environment
- Install Python libraries
```bash
pip install -r requirements.txt
```
- [Download](https://github.com/codyfarlow1/CS898BA-HWOne) Homework Image and Make sure the name is <b>"HW1_IMG_CS898BA.png"</b>

# Run code
- Execute the Python script
```bash
python -u KhangDuong_HW1.py
```

---

# Output Examples

![HW1_IMG_CS898BA_binary_blurred_sigma1.5_plot.png](plots/HW1_IMG_CS898BA_binary_blurred_sigma1.5_plot.png)

![HW1_IMG_CS898BA_HLS_blurred_sigma1.5_plot.png](plots/HW1_IMG_CS898BA_HLS_blurred_sigma1.5_plot.png)

![HW1_IMG_CS898BA_binary_transformed_2_blurred_sigma3.5_plot.png](plots/HW1_IMG_CS898BA_binary_transformed_2_blurred_sigma3.5_plot.png)

![HW1_IMG_CS898BA_binary_transformed_1_plot.png](plots/HW1_IMG_CS898BA_binary_transformed_1_plot.png)

![HW1_IMG_CS898BA_equalized_transformed_1_blurred_sigma3.0_plot.png](plots/HW1_IMG_CS898BA_equalized_transformed_1_blurred_sigma3.0_plot.png)

![HW1_IMG_CS898BA_CIELAB_blurred_sigma0.5_plot.png](plots/HW1_IMG_CS898BA_CIELAB_blurred_sigma0.5_plot.png)

---

# Discussions

### Effect of Gaussian blur σ (Part 2.8)
Low sigma values like 0.5 and 1.0 helps in smoothing the images while preserving most of the details.
As the sigma value increases, the images lose more and more details like edges and textures, which makes it harder to detect edges later on.

### Edge detection — pros, cons, and which wins for this set (Part 3.5)

**Sobel**
- *Pros:* Details edges in both horizontal and vertical directions.
- *Cons:* May produce thick and/or non-connected edges, sensitive to noise.

**Laplacian**
- *Pros:* Detects edges in all directions, good for finding fine details.
- *Cons:* Very sensitive to noise, may produce false edges.

**Canny**
- *Pros:* Good for detecting edges in noisy images, provides thin and connected edges, uses non-maximum suppression.
- *Cons:* Requires tuning of threshold cutoffs, may miss weak edges or produce false edges.

**Prewitt**
- *Pros:* Simple and fast, good for detecting edges in clean images.
- *Cons:* May produce thicker edges, sensitive to noise.

For this image set, Sobel is the best because it provides clear edges out of the most images, while Laplacian and Prewitt produce more noise and Canny misses some edges in the blurred images. However, for bright images, Prewitt can perform better than Sobel because Sobel produce too many edges.

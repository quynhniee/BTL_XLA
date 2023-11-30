import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import numpy as np

IMG_WIDTH = 500

######## Conditions functions ##########

def is_valid_kernel_size(input_text):
    """Kiểm tra xem giá trị nhập vào có phải là số nguyên lẻ không."""
    try:
        kernel_size = int(input_text)
        return kernel_size > 0 and kernel_size % 2 != 0
    except ValueError:
        return False

def is_valid_k_value(input_text):
    """Kiểm tra xem giá trị nhập vào có phải là số nguyên lẻ không."""
    try:
        k_value = int(input_text)
        return k_value > 0 and k_value % 2 != 0
    except ValueError:
        return False

######## Conditions functions ##########

######## Common functions #########
def resize_image(image, target_width):
    """Chỉnh kích thước ảnh với chiều rộng mong muốn."""
    aspect_ratio = image.shape[1] / image.shape[0]
    target_height = int(target_width / aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image

def apply_filter(image, filter_function):
    """Áp dụng bộ lọc cho ảnh và trả về kết quả."""
    return filter_function(image)

def open_file():
    """Mở hộp thoại để chọn một tệp tin ảnh."""
    file_path = filedialog.askopenfilename()
    if file_path:
        load_and_display_original(file_path)

def load_and_display_original(file_path):
    """Load và hiển thị ảnh gốc trong cửa sổ."""
    global original_image
    original_image = cv2.imread(file_path)
    resized_original = resize_image(original_image, target_width=IMG_WIDTH)
    show_images(resized_original)

def show_images(original_image):
    """Hiển thị ảnh gốc và ảnh sau khi chỉnh sửa."""
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)

    # Hiển thị ảnh gốc
    label_original.config(image=img)
    label_original.image = img

    # Hiển thị ảnh sau khi chỉnh sửa 
    label_negative.config(image=None)
    label_negative.image = None

    # Hiển thị button Apply Filter
    button_apply_filter.pack(pady=10)

######## Common functions #########


######## Algorithm functions #########

def apply_negative(image):
    """Áp dụng âm bản lên ảnh."""
    return 255 - image

# --------------------------------

def apply_grayscale(image):
    """Chuyển ảnh sang dạng grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --------------------------------

def apply_threshold(image, threshold_value):
    """Áp dụng bộ lọc phân ngưỡng cho ảnh."""
    _, thresh_binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh_binary

# Hàm áp dụng bộ lọc phân ngưỡng và hiển thị
def apply_threshold_and_display():
    global original_image
    if original_image is not None:
        threshold_value = simpledialog.askfloat("Threshold Value", "Enter threshold value:")
        if threshold_value is not None:
            thresholded_image = apply_threshold(original_image, threshold_value)
            thresholded_img = resize_image(thresholded_image, target_width=IMG_WIDTH)
            thresholded_img = cv2.cvtColor(thresholded_img, cv2.COLOR_BGR2RGB)
            thresholded_img = Image.fromarray(thresholded_img)
            thresholded_img = ImageTk.PhotoImage(thresholded_img)

            # Hiển thị ảnh sau khi chỉnh sửa
            label_negative.config(image=thresholded_img)
            label_negative.image = thresholded_img

# --------------------------------

def apply_weighted_mean(image, kernel_size):
    """Áp dụng bộ lọc trung bình có trọng số cho ảnh."""
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    result = cv2.filter2D(image, -1, kernel)
    return result

# Hàm áp dụng bộ lọc trung bình có trọng số và hiển thị
def apply_weighted_mean_and_display():
    global original_image
    if original_image is not None:
        kernel_size = simpledialog.askinteger("Kernel Size", "Enter kernel size (odd number):")
        if kernel_size is not None and is_valid_kernel_size(kernel_size):
            weighted_mean_filtered_image = apply_weighted_mean(original_image, kernel_size)
            weighted_mean_filtered_img = resize_image(weighted_mean_filtered_image, target_width=IMG_WIDTH)
            weighted_mean_filtered_img = cv2.cvtColor(weighted_mean_filtered_img, cv2.COLOR_BGR2RGB)
            weighted_mean_filtered_img = Image.fromarray(weighted_mean_filtered_img)
            weighted_mean_filtered_img = ImageTk.PhotoImage(weighted_mean_filtered_img)

            # Hiển thị ảnh sau khi chỉnh sửa
            label_negative.config(image=weighted_mean_filtered_img)
            label_negative.image = weighted_mean_filtered_img

# --------------------------------

def apply_k_nearest_mean(image, k, threshold):
    """Áp dụng bộ lọc k giá trị gần nhất cho ảnh."""
    result = np.zeros_like(image, dtype=np.float32)

    rows, cols = image.shape[:2]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = image[i - 1:i + 2, j - 1:j + 2].flatten()
            window = np.sort(window)
            window_size = len(window)

            # Tính trung bình cho k giá trị gần nhất
            mean_value = np.sum(window[:k]) / k

            # Kiểm tra ngưỡng
            if np.abs(image[i, j] - mean_value) > threshold:
                result[i, j] = image[i, j]
            else:
                result[i, j] = mean_value

    return np.uint8(result)

# Hàm áp dụng bộ lọc k giá trị gần nhất và hiển thị
def apply_k_nearest_mean_filter_and_display():
    global original_image
    if original_image is not None:
        k_value = simpledialog.askinteger("k-Nearest Mean Filter", "Enter k value (odd number):")
        if k_value is not None and is_valid_k_value(k_value):
            threshold_value = simpledialog.askinteger("Threshold", "Enter threshold value:")
            if threshold_value is not None:
                k_nearest_mean_filtered_image = apply_k_nearest_mean(original_image, k_value, threshold_value)
                k_nearest_mean_filtered_img = resize_image(k_nearest_mean_filtered_image, target_width=IMG_WIDTH)
                k_nearest_mean_filtered_img = cv2.cvtColor(k_nearest_mean_filtered_img, cv2.COLOR_BGR2RGB)
                k_nearest_mean_filtered_img = Image.fromarray(k_nearest_mean_filtered_img)
                k_nearest_mean_filtered_img = ImageTk.PhotoImage(k_nearest_mean_filtered_img)

                # Hiển thị ảnh sau khi chỉnh sửa
                label_negative.config(image=k_nearest_mean_filtered_img)
                label_negative.image = k_nearest_mean_filtered_img

# --------------------------------

def median_filter(image, kernel_size):
    result = np.zeros_like(image)

    rows, cols = image.shape[:2]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = image[i - 1:i + 2, j - 1:j + 2].flatten()
            median_value = np.median(window)
            result[i, j] = median_value

    return np.uint8(result)

# Hàm áp dụng bộ lọc trung vị
def apply_median_filter_and_display():
    global original_image
    if original_image is not None:
        kernel_size = simpledialog.askinteger("Median Filter", "Enter kernel size:")
        if kernel_size is not None:
            median_filtered_image = median_filter(original_image, kernel_size)
            median_filtered_img = resize_image(median_filtered_image, target_width=IMG_WIDTH)
            median_filtered_img = cv2.cvtColor(median_filtered_img, cv2.COLOR_BGR2RGB)
            median_filtered_img = Image.fromarray(median_filtered_img)
            median_filtered_img = ImageTk.PhotoImage(median_filtered_img)

            # Hiển thị ảnh sau khi chỉnh sửa
            label_negative.config(image=median_filtered_img)
            label_negative.image = median_filtered_img

# --------------------------------

# Hàm áp dụng cân bằng lược đồ xám
def apply_histogram_equalization(image):
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tính lược đồ xám
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Tính tổng tích lũy
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Tìm hàm biến đổi s = T(r)
    lut = np.interp(image.flatten(), bins[:-1], cdf_normalized)

    # Reshape lại kích thước ảnh và đưa về kiểu dữ liệu uint8
    result = np.uint8(lut.reshape(image.shape))

    return result

# --------------------------------

def apply_filter_and_display(filter_function):
    """Áp dụng bộ lọc và hiển thị kết quả."""
    global original_image
    if original_image is not None:
        filtered_image = apply_filter(original_image, filter_function)
        filtered_img = resize_image(filtered_image, target_width=IMG_WIDTH)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
        filtered_img = Image.fromarray(filtered_img)
        filtered_img = ImageTk.PhotoImage(filtered_img)

        # Hiển thị ảnh sau khi chỉnh sửa
        label_negative.config(image=filtered_img)
        label_negative.image = filtered_img

######## Algorithm functions #########


# Tạo cửa sổ tkinter
root = tk.Tk()
root.title("Image Filter")
root.geometry("1200x600")

# Tạo thanh cuộn
scrollbar = tk.Scrollbar(root, orient="vertical")

# Tạo canvas để chứa nội dung và kết nối với thanh cuộn
canvas = tk.Canvas(root, yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)

# Kết nối thanh cuộn với canvas
scrollbar.config(command=canvas.yview)
scrollbar.pack(side="right", fill="y")

# Tạo frame bên trong canvas để chứa nội dung
frame_inside_canvas = tk.Frame(canvas)


############ Build layout #############

# Tạo nút và hộp chọn tệp tin
button_open = tk.Button(frame_inside_canvas, text="Open Image", command=open_file)
button_open.pack(pady=10)

# Tạo dòng chứa 2 cột
frame_row = tk.Frame(frame_inside_canvas)
frame_row.pack()

# Cột chứa label ảnh gốc
label_original = tk.Label(frame_row, text="Original Image")
label_original.pack(side=tk.LEFT, padx=10)

# Cột chứa label ảnh sau khi chỉnh sửa
label_negative = tk.Label(frame_row, text="Edited Image")
label_negative.pack(side=tk.LEFT, padx=10)

# Tạo nút áp dụng bộ lọc âm bản 
button_apply_filter = tk.Button(frame_inside_canvas, text="Apply Negative Filter", command=lambda: apply_filter_and_display(apply_negative))
button_apply_filter.pack(pady=10)

# Tạo nút áp dụng bộ lọc chuyển sang grayscale 
button_apply_grayscale = tk.Button(frame_inside_canvas, text="Apply Grayscale Filter", command=lambda: apply_filter_and_display(apply_grayscale))
button_apply_grayscale.pack(pady=10)

# Tạo nút áp dụng bộ lọc phân ngưỡng 
button_apply_threshold = tk.Button(frame_inside_canvas, text="Apply Threshold Filter", command=apply_threshold_and_display)
button_apply_threshold.pack(pady=10)

# Tạo nút áp dụng bộ lọc cân bằng lược đồ xám 
button_apply_histogram_equalization = tk.Button(frame_inside_canvas, text="Apply Histogram Equalization", command=lambda: apply_filter_and_display(apply_histogram_equalization))
button_apply_histogram_equalization.pack(pady=10)

# Tạo nút áp dụng bộ lọc trung bình có trọng số 
button_apply_weighted_mean = tk.Button(frame_inside_canvas, text="Apply Weighted Mean Filter", command=apply_weighted_mean_and_display)
button_apply_weighted_mean.pack(pady=10)

# Tạo nút áp dụng bộ lọc k giá trị gần nhất (đang lỗi)
button_apply_k_nearest_mean_filter = tk.Button(frame_inside_canvas, text="Apply k-Nearest Mean Filter", command=apply_k_nearest_mean_filter_and_display)
button_apply_k_nearest_mean_filter.pack(pady=10)

# Tạo nút áp dụng bộ lọc trung vị 
button_apply_median_filter = tk.Button(frame_inside_canvas, text="Apply Median Filter", command=apply_median_filter_and_display)
button_apply_median_filter.pack(pady=10)

############ Build layout #############


# Đặt nội dung của canvas là frame_inside_canvas
canvas.create_window((0, 0), window=frame_inside_canvas, anchor="nw")

# Bắt sự kiện thay đổi kích thước của canvas
def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

canvas.bind("<Configure>", on_configure)

# Chạy giao diện
root.mainloop()

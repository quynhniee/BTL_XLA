import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import numpy as np

IMG_WIDTH = 500
PAD_Y = 10
PAD_X = 10

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
    label_filtered.config(image=None)
    label_filtered.image = None


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
        label_filtered.config(image=filtered_img)
        label_filtered.image = filtered_img

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
            label_filtered.config(image=thresholded_img)
            label_filtered.image = thresholded_img

# --------------------------------

def logarithmic_transformation(image, c=1):
    """ Áp dụng biến đổi logarithmic cho ảnh."""
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng biến đổi logarithmic
    result = c * np.log1p(image)

    # Chuẩn hóa giá trị pixel về khoảng [0, 255]
    result = ((result - np.min(result)) / (np.max(result) - np.min(result))) * 255.0

    # Chuyển đổi kiểu dữ liệu về uint8
    result = np.uint8(result)

    return result

def apply_logarithmic_transformation_and_display():
    global original_image
    if original_image is not None:
        c_value = simpledialog.askfloat("Logarithmic Transformation", "Enter the constant 'c' value:")
        if c_value is not None:
            log_transformed_image = logarithmic_transformation(original_image, c=c_value)
            log_transformed_img = resize_image(log_transformed_image, target_width=IMG_WIDTH)
            log_transformed_img = cv2.cvtColor(log_transformed_img, cv2.COLOR_BGR2RGB)
            log_transformed_img = Image.fromarray(log_transformed_img)
            log_transformed_img = ImageTk.PhotoImage(log_transformed_img)

            # Hiển thị ảnh sau khi chỉnh sửa
            label_filtered.config(image=log_transformed_img)
            label_filtered.image = log_transformed_img

# --------------------------------

def power_law_transformation(image, gamma=1, c=1):
    """Áp dụng biến đổi hàm mũ cho ảnh."""
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng biến đổi hàm mũ
    result = c * np.power(image, gamma)

    # Chuẩn hóa giá trị pixel về khoảng [0, 255]
    result = ((result - np.min(result)) / (np.max(result) - np.min(result))) * 255.0

    # Chuyển đổi kiểu dữ liệu về uint8
    result = np.uint8(result)

    return result

def apply_power_law_transformation_and_display():
    global original_image
    if original_image is not None:
        gamma_value = simpledialog.askfloat("Power Law Transformation", "Enter the gamma value:")
        if gamma_value is not None:
            c_value = simpledialog.askfloat("Power Law Transformation", "Enter the constant 'c' value:")
            if c_value is not None:
                power_law_transformed_image = power_law_transformation(original_image, gamma=gamma_value, c=c_value)
                power_law_transformed_img = resize_image(power_law_transformed_image, target_width=IMG_WIDTH)
                power_law_transformed_img = cv2.cvtColor(power_law_transformed_img, cv2.COLOR_BGR2RGB)
                power_law_transformed_img = Image.fromarray(power_law_transformed_img)
                power_law_transformed_img = ImageTk.PhotoImage(power_law_transformed_img)

                # Hiển thị ảnh sau khi chỉnh sửa
                label_filtered.config(image=power_law_transformed_img)
                label_filtered.image = power_law_transformed_img

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
            label_filtered.config(image=weighted_mean_filtered_img)
            label_filtered.image = weighted_mean_filtered_img

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
            if np.any(np.abs(image[i, j] - mean_value) > threshold):
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
                label_filtered.config(image=k_nearest_mean_filtered_img)
                label_filtered.image = k_nearest_mean_filtered_img

# --------------------------------

def median_filter(image, kernel_size):
    """Áp dụng bộ lọc trung vị cho ảnh."""
    height, width = image.shape[:2]
    result = np.zeros_like(image)

    half_kernel = kernel_size // 2

    for i in range(half_kernel, height - half_kernel):
        for j in range(half_kernel, width - half_kernel):
            window = image[i - half_kernel:i + half_kernel + 1, j - half_kernel:j + half_kernel + 1]
            result[i, j] = np.median(window)

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
            label_filtered.config(image=median_filtered_img)
            label_filtered.image = median_filtered_img

# --------------------------------

# Hàm áp dụng cân bằng lược đồ xám
def apply_histogram_equalization(image):
    """Áp dụng bộ lọc cân bằng lược đồ xám cho ảnh."""
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

def apply_roberts_operator(image):
    """Áp dụng toán tử Roberts cho ảnh."""
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo kernel cho toán tử Roberts theo chiều ngang-dọc
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    # Áp dụng toán tử Roberts theo chiều ngang-dọc
    result_x = cv2.filter2D(image, -1, kernel_x)
    result_y = cv2.filter2D(image, -1, kernel_y)

    # Tính toán độ lớn của đạo hàm
    result = np.sqrt(np.square(result_x) + np.square(result_y))

    result *= 255
    # Chuyển đổi kiểu dữ liệu về uint8
    result = np.uint8(result)

    return result

# --------------------------------

def apply_prewitt_operator(image):
    """Áp dụng toán tử Prewitt cho ảnh."""
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo kernel cho toán tử Prewitt
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Áp dụng toán tử Prewitt
    result_x = cv2.filter2D(image, -1, kernel_x)
    result_y = cv2.filter2D(image, -1, kernel_y)

    # Tính toán tổng hợp của kết quả
    result = np.sqrt(np.square(result_x) + np.square(result_y))

    result *= 255
    # Chuyển đổi kiểu dữ liệu về uint8
    result = np.uint8(result)

    return result

# --------------------------------

def apply_sobel_operator(image):
    """Áp dụng toán tử Sobel cho ảnh."""
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo kernel cho toán tử Sobel
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Áp dụng toán tử Sobel
    result_x = cv2.filter2D(image, -1, kernel_x)
    result_y = cv2.filter2D(image, -1, kernel_y)

    # Tính toán tổng hợp của kết quả
    result = np.sqrt(np.square(result_x) + np.square(result_y))
    result *= 255

    # Chuyển đổi kiểu dữ liệu về uint8
    result = np.uint8(result)

    return result

# --------------------------------

def apply_laplacian_operator(image):
    """Áp dụng toán tử Laplacian cho ảnh."""
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo kernel cho toán tử Laplacian
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Áp dụng toán tử Laplacian
    result = cv2.filter2D(image, -1, kernel)

    # Chuyển đổi kiểu dữ liệu về uint8
    result = np.uint8(result)

    return result

# --------------------------------

def canny_algorithm(image, low_threshold, high_threshold):
    """Áp dụng thuật toán Canny cho ảnh."""
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng thuật toán Canny
    result = cv2.Canny(image, low_threshold, high_threshold)

    return result

def apply_canny_algorithm_and_display():
    global original_image
    if original_image is not None:
        # Giá trị ngưỡng cao và thấp
        low_threshold = simpledialog.askinteger("Canny", "Low Threshold:")
        high_threshold = simpledialog.askinteger("Canny", "High Threshold:")
        if low_threshold is not None and high_threshold is not None :
            canny_image = canny_algorithm(original_image, low_threshold, high_threshold)
            canny_img = resize_image(canny_image, target_width=IMG_WIDTH)
            canny_img = cv2.cvtColor(canny_img, cv2.COLOR_BGR2RGB)
            canny_img = Image.fromarray(canny_img)
            canny_img = ImageTk.PhotoImage(canny_img)

            # Hiển thị ảnh sau khi chỉnh sửa
            label_filtered.config(image=canny_img)
            label_filtered.image = canny_img

# --------------------------------

def apply_otsu_algorithm(image):
    """Áp dụng thuật toán Otsu cho ảnh."""
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng thuật toán Otsu
    _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return result

# --------------------------------

def erosion_operation(image, kernel_size, iterations=1):
    """Áp dụng phép co cho ảnh."""
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo kernel cho phép co
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Áp dụng phép co
    result = cv2.erode(image, kernel, iterations)

    return result

def apply_erosion_operation_and_display():
    global original_image
    if original_image is not None:
        erosion_kernel_size = simpledialog.askinteger("Erosion", "Erosion Kernel Size:")
        erosion_interations = simpledialog.askinteger("Erosion", "Erosion iterations:")
        if erosion_kernel_size is not None and erosion_interations is not None :
            erosion_image = erosion_operation(original_image, erosion_kernel_size, erosion_interations)
            erosion_img = resize_image(erosion_image, target_width=IMG_WIDTH)
            erosion_img = cv2.cvtColor(erosion_img, cv2.COLOR_BGR2RGB)
            erosion_img = Image.fromarray(erosion_img)
            erosion_img = ImageTk.PhotoImage(erosion_img)

            # Hiển thị ảnh sau khi chỉnh sửa
            label_filtered.config(image=erosion_img)
            label_filtered.image = erosion_img

# --------------------------------

def dilation_operation(image, kernel_size, iterations):
    """Áp dụng phép giãn cho ảnh."""
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo kernel cho phép giãn hình
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Áp dụng phép giãn hình
    result = cv2.dilate(image, kernel, iterations=iterations)

    return result

def apply_dilation_operation_and_display():
    global original_image
    if original_image is not None:
        kernel_size = simpledialog.askinteger("Dilation", "Enter Dilation Kernel Size:")
        iterations = simpledialog.askinteger("Dilation", "Enter Dilation Iterations:")
        if kernel_size is not None and iterations is not None :
            dilation_image = dilation_operation(original_image, kernel_size, iterations)
            dilation_img = resize_image(dilation_image, target_width=IMG_WIDTH)
            dilation_img = cv2.cvtColor(dilation_img, cv2.COLOR_BGR2RGB)
            dilation_img = Image.fromarray(dilation_img)
            dilation_img = ImageTk.PhotoImage(dilation_img)

            # Hiển thị ảnh sau khi chỉnh sửa
            label_filtered.config(image=dilation_img)
            label_filtered.image = dilation_img

# --------------------------------

def opening_operation(image, kernel_size):
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo kernel cho phép mở
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Áp dụng phép mở
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return result

def apply_opening_operation_and_display():
    global original_image
    if original_image is not None:
        kernel_size = simpledialog.askinteger("Opening", "Enter Opening Kernel Size:")
        if kernel_size is not None:
            opening_image = opening_operation(original_image, kernel_size)
            opening_img = resize_image(opening_image, target_width=IMG_WIDTH)
            opening_img = cv2.cvtColor(opening_img, cv2.COLOR_BGR2RGB)
            opening_img = Image.fromarray(opening_img)
            opening_img = ImageTk.PhotoImage(opening_img)

            # Hiển thị ảnh sau khi chỉnh sửa
            label_filtered.config(image=opening_img)
            label_filtered.image = opening_img

# --------------------------------

def closing_operation(image, kernel_size):
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo kernel cho phép đóng
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Áp dụng phép đóng
    result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return result

def apply_closing_operation_and_display():
    global original_image
    if original_image is not None:
        kernel_size = simpledialog.askinteger("Closing", "Enter Closing Kernel Size:")
        if kernel_size is not None:
            closing_image = closing_operation(original_image, kernel_size)
            closing_img = resize_image(closing_image, target_width=IMG_WIDTH)
            closing_img = cv2.cvtColor(closing_img, cv2.COLOR_BGR2RGB)
            closing_img = Image.fromarray(closing_img)
            closing_img = ImageTk.PhotoImage(closing_img)

            # Hiển thị ảnh sau khi chỉnh sửa
            label_filtered.config(image=closing_img)
            label_filtered.image = closing_img


######## Algorithm functions #########


# Tạo cửa sổ tkinter
root = tk.Tk()
root.title("Image Filter")
root.geometry("1200x1000")

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
button_open.grid(row=0, column=0, padx=PAD_X, pady=PAD_Y)
tk.Label(frame_inside_canvas, text="* Resize window để xuất hiện scrollbar").grid(row=0, column=1, padx=PAD_X, pady=PAD_Y)

# Cột chứa label ảnh gốc
label_original = tk.Label(frame_inside_canvas, text="Original Image")
label_original.grid(row=1, column=0, padx=PAD_X, pady=PAD_Y)

# Cột chứa label ảnh sau khi chỉnh sửa
label_filtered = tk.Label(frame_inside_canvas, text="Edited Image")
label_filtered.grid(row=1, column=1, padx=PAD_X, pady=PAD_Y)

# Mở sẵn 1 ảnh
load_and_display_original("backend/python/oggy.jpeg")

# List button
button_info_list = [
    ("Apply Negative Filter", lambda: apply_filter_and_display(apply_negative)),
    ("Apply Grayscale Filter", lambda: apply_filter_and_display(apply_grayscale)),
    ("Apply Threshold Filter", apply_threshold_and_display),
    ("Apply Logarithmic Transformation", apply_logarithmic_transformation_and_display),
    ("Apply Power Law Transformation",  apply_power_law_transformation_and_display),
    ("Apply Histogram Equalization", lambda: apply_filter_and_display(apply_histogram_equalization)),
    ("Apply Weighted Mean Filter", apply_weighted_mean_and_display),
    ("Apply k-Nearest Mean Filter", apply_k_nearest_mean_filter_and_display),
    ("Apply Median Filter", apply_median_filter_and_display),
    ("Apply Roberts Operator", lambda: apply_filter_and_display(apply_roberts_operator)),
    ("Apply Prewitt Operator", lambda: apply_filter_and_display(apply_prewitt_operator)),
    ("Apply Sobel Operator", lambda: apply_filter_and_display(apply_sobel_operator)),
    ("Apply Laplacian Operator", lambda: apply_filter_and_display(apply_laplacian_operator)),
    ("Apply Canny Algorithm", apply_canny_algorithm_and_display),
    ("Apply Otsu Algorithm", lambda: apply_filter_and_display(apply_otsu_algorithm)),
    ("Apply Erosion Operation", apply_erosion_operation_and_display),
    ("Apply Dilation Operation", apply_dilation_operation_and_display),
    ("Apply Opening Operation", apply_opening_operation_and_display),
    ("Apply Closing Operation", apply_closing_operation_and_display)
]

num_columns = 2
for i, (button_text, command) in enumerate(button_info_list):
    row = 3 + i // num_columns
    column = i % num_columns
    button_apply_filter = tk.Button(frame_inside_canvas, text=button_text, command=command)
    button_apply_filter.grid(row=row, column=column, padx=PAD_X, pady=PAD_Y)

############ Build layout #############


# Đặt nội dung của canvas là frame_inside_canvas
canvas.create_window((0, 0), window=frame_inside_canvas, anchor="nw")

# Bắt sự kiện thay đổi kích thước của canvas
def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

canvas.bind("<Configure>", on_configure)

# Chạy giao diện
root.mainloop()

import cv2
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
import easyocr
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
    
# """Parameter"""
# batch_size = 5
# eps_pixel = 15 
# padding = 20 
# thresh_intensity = 20
# n_roi_rows = 5
# batch_progress_bar = True

reader = easyocr.Reader(['ko','en']) # this needs to run only once to load the model into memory

"""Load multi files"""
# local
def load_files_local(file_paths):
    name_images = {}
    for file_path in file_paths:
        """Load file"""
        # (file_path) -> (file_name,)
        file_name = file_path.split("/")[-1]
        file_format = file_name.lower().split(".")[-1]
        rgb_images = []
        if file_format == "png":
            image =  cv2.imread(file_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb_image)
        elif file_format == "pdf":
            pages = convert_from_path(file_path)
            for i, page in enumerate(pages[:]): 
                rgb_image = np.array(page)
                rgb_images.append(rgb_image)
        else:
            raise
        name_images[file_name] = rgb_images
    return name_images


# colab
def load_files_colab():
    from google.colab import files
    from io import BytesIO
    from PIL import Image

    uploaded = files.upload()
    name_images = {}
    for file_name, value in uploaded.items():
        # Load file 
        file_format = file_name.lower().split(".")[-1]
        rgb_images = []
        if file_format in ["png", "jpg"]:
            image = Image.open(BytesIO(uploaded[file_name]))
            rgba_image = np.array(image)
            rgb_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2RGB)
            rgb_images.append(rgb_image)
        elif file_format == "pdf":
            pages = convert_from_bytes(uploaded[file_name])
            for i, page in enumerate(pages): 
                rgb_image = np.array(page)
                rgb_images.append(rgb_image)
        else:
            raise ValueError("'{}' format is not supported. Only ['png', 'pdf'] are supported.".format(format))
        name_images[file_name] = rgb_images
    return name_images


"""Preprocessing"""
def generate_feat_image(rgb_image, thresh_intensity):
    # convert into gray image
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(gray_image).shape[1]//100
    # Defining a kernel to detect any black character
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, kernel_len))
    # Use kernel to detect any black character 
    filtered_image = cv2.erode(gray_image, kernel, iterations=3)
    horizontal_lines = cv2.dilate(filtered_image, kernel, iterations=3)
    # use threshold to detect black characters
    thresh, feat_image = cv2.threshold(src=filtered_image, thresh=thresh_intensity, maxval=255, type=cv2.THRESH_BINARY) 
    # inverting the image color : white <-> black
    feat_image = 255-feat_image 
    return feat_image


"""Find the table outline by a simple rule"""
def gererate_roi(feat_image, rgb_image, file_format, padding):
    if file_format == "png":
        # horizon region of interest
        hor_position = np.where(feat_image.sum(axis=0) > 0)[0]
        # vertical region of interest
        coi_ind = np.argmax(feat_image.sum(axis=0))
        coi = feat_image[:, coi_ind] # column of interest
        ver_position = np.where(coi > 0)[0] 
        # crop roi
        feat_roi = feat_image[ver_position[0]-padding:, 
                    hor_position[0]-padding:hor_position[-1]+padding]
                    
        rgb_roi = rgb_image[ver_position[0]-padding:, 
                        hor_position[0]-padding:hor_position[-1]+padding, :]
    elif file_format == "pdf":
        # crop head line
        hor_sum = feat_image.sum(axis=1) > 0
        hor_sum_bin = hor_sum.astype(int)
        ver_something = np.where((hor_sum_bin[1:] - hor_sum_bin[:-1]) == 1)[0]
        feat_image_crop = feat_image[ver_something[1]-padding:, :]
        rgb_image_crop = rgb_image[ver_something[1]-padding:, :]
        # vertical region of interest
        coi_ind = np.argmax(feat_image_crop.sum(axis=0))
        coi = feat_image_crop[:, coi_ind] # column of interest
        ver_position = np.where(coi > 0)[0] 
        # horizon region of interest
        horizontal_roi = feat_image_crop.sum(axis=0) > 0
        horizontal_roi = horizontal_roi.astype(int)
        hor_position = np.where((horizontal_roi[1:] - horizontal_roi[:-1]) == 1)[0]
        # crop roi
        padding = 10
        feat_roi = feat_image_crop[ver_position[0]-padding:, 
                                hor_position[0]-padding:]
                    
        rgb_roi = rgb_image_crop[ver_position[0]-padding:, 
                                hor_position[0]-padding:, 
                                :]
    else :
        raise ValueError("'{}' format is not supported. Only ['png', 'pdf'] are supported.".format(file_format))
    return feat_roi, rgb_roi


"""ocr"""
def post_process(row):
    if len(row) == 10:
        row = row[1:]
    mapper = {
        "0":["o", "O", "U", "D", "C"],
        "1":["i", "l", "I", "t"], 
        "2":["z", "Z"],
        "4":["-", "A"],
        "5":["s", "S"],
        "7":["T"],
        "8":["G", "E", "B", "e", "@"],
        "9":["g"],
        "17":["V"],
        "h":["+"],
        "m":["r"],
    }
    for i in range(1, len(row)):
        for key, values in mapper.items():
            for value in values:
                row[i] = row[i].replace(value, key)
    return row

def row_to_texts(row, eps_pixel):
    texts = []
    prev_rx = -999 # right x position
    prev_text = ""
    for position, text in row:
        if position[0][0] - prev_rx > eps_pixel: # left x - prev_rx
            texts.append(prev_text)
            prev_text = text
            prev_rx = position[1][0] # right x position
        else:
            prev_text = text if prev_text == "" else (prev_text + " " + text)
    texts.append(prev_text) # last text
    texts.pop(0) # first empty string
    return texts

def ocr(image, eps_pixel):
    result = reader.readtext(image)
    result = sorted(result, key=lambda x:x[0][0][1]) # by top y position
    row = []
    rows = []
    prev_ty = -999
    for position, text, _ in result:
        if abs(prev_ty - position[0][1]) < eps_pixel:
            row.append((position, text))
        else:
            sorted_row = sorted(row, key=lambda x:x[0][0][0]) # by left x position
            row_texts = row_to_texts(sorted_row, eps_pixel)
            row_pp = post_process(row_texts)
            rows.append(row_pp)
            prev_ty = position[0][1] # previous top y position
            row = [(position, text)]
    rows.pop(0) # 첫번째 빈 list
    sorted_row = sorted(row, key=lambda x:x[0][0][0]) # by left x position
    row_texts = [_[1] for _ in sorted_row]
    row_pp = post_process(row_texts)
    rows.append(row_pp) # 마지막 row
    return rows


"""find rows"""
def find_end_of_rows(feat_roi):
    vertical_roi2 = feat_roi.sum(axis=1) > 0
    vertical_roi2 = vertical_roi2.astype(int)
    end_of_rows = np.where((vertical_roi2[1:] - vertical_roi2[:-1]) == -1)[0]
    end_of_rows = [0] + list(end_of_rows)
    return end_of_rows


"""batch process"""
def ocr_by_batched_row(feat_roi, rgb_roi, padding, n_roi_rows, eps_pixel, progress_bar=False):
    end_of_rows = find_end_of_rows(feat_roi)

    n_steps = len(end_of_rows) // n_roi_rows 
    n_steps = n_steps + 1 if len(end_of_rows) % n_roi_rows else n_steps

    total = []
    a = None
    b = -padding
    if progress_bar is True:
        steps = tqdm(range(1, n_steps))
    else:
        steps = range(1, n_steps)
    for step in steps:
        a = b
        b = end_of_rows[step*n_roi_rows]
        crop_image1 = rgb_roi[a+padding:b+padding, : ,:]
        rows = ocr(crop_image1, eps_pixel)
        total = total + rows
        # plt.imshow(crop_image1)
        # plt.show()
        # print([(row[0], len(row)) for row in rows if row])
    else:
        crop_image1 = rgb_roi[b+padding:, : ,:]
        rows = ocr(crop_image1, eps_pixel)
        total = total + rows
        # plt.imshow(crop_image1)
        # plt.show()
        # print([(row[0], len(row)) for row in rows if row])
    total = [row for row in total if len(row)>4]
    return total

def total_to_df(total):
    df = pd.DataFrame(total)
    df = df.rename(columns={0:"이름", 1:"계", 2:"월", 3:"화", 4:"수", 5:"목", 6:"금", 7:"토", 8:"일"})
    return df


"""Extract table"""
def extract_table_from_image(
    rgb_image,
    file_format,
    batch_size = 5,
    eps_pixel = 15 ,
    padding = 20 ,
    thresh_intensity = 20,
    n_roi_rows=5,
    batch_progress_bar=False,
    ):
    """
    Extract a table from a image with batch processing.
    Each batch is a cropped image so that rows are batched.

    Args:
        rgb_image (numpy.array) : a rgb_image to be processed.
    Return:
        df (pandas.DataFrame) : the extracted table 
    """
    feat_image = generate_feat_image(rgb_image, thresh_intensity)
    try:
        feat_roi, rgb_roi = gererate_roi(feat_image, rgb_image, file_format, padding)
    except IndexError:
        return pd.DataFrame([])
    total = ocr_by_batched_row(feat_roi, rgb_roi, padding=padding, n_roi_rows=n_roi_rows, eps_pixel=eps_pixel, progress_bar=batch_progress_bar)
    df = total_to_df(total) # list -> df
    return df


import logging

import cv2
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def create_logger():
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:- %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("APT_Realignment")
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def get_countours(mask_image):
    # get countours
    edged = cv2.Canny(mask_image, 0.5, 1)
    # Finding Contours
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def draw_countours(input_image, mask_image):
    output_image = input_image.copy()
    contours = get_countours(mask_image=mask_image)
    for contour in contours:
        cv2.drawContours(output_image, contour, -1, (0, 255, 0), 3)
    return output_image


def split_dataset(rgb_images, mask_images, train_size=0.80, test_size=0.20):
    rgb_images = sorted(rgb_images)
    mask_images = sorted(mask_images)
    logger = create_logger()
    data = np.asarray([(rgb, mask) for rgb, mask in zip(rgb_images, mask_images)])
    x_train, x_test = train_test_split(data, train_size=train_size, test_size=test_size)
    logger.info('Total Training RGB images:{}'.format(x_train.shape[0]))
    logger.info('Total Test RGB images:{}'.format(x_test.shape[0]))
    return x_train, x_test


def overlay_mask(image, mask_color, alpha=0.4):
    ret = image.copy()
    ret[mask_color.sum(axis=-1) > 0] = mask_color[mask_color.sum(axis=-1) > 0]
    ret = cv2.addWeighted(ret, alpha, image, 1 - alpha, 0)
    return ret


def decode_segmentation_masks_rgb(mask, color_map=[125, 32, 128], n_classes=1):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    # for l in range(0, n_classes):
    idx = mask == 1
    r[idx] = color_map[0]
    g[idx] = color_map[1]
    b[idx] = color_map[2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def decode_segmentation_masks_gray(mask):
    gray = np.zeros_like(mask).astype(np.uint8)
    idx = mask > 0.5  # since sigmoid function is used in binary model
    gray[idx] = 255
    return gray


def read_image_mask(path, mask=False):
    if mask:
        img = skimage.io.imread(path, plugin='tifffile')
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
        return img
    else:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_sample_image(test_images, target_shape=(512, 512, 3), image_label='test', mask_label='test_labels'):
    image_name = test_images[np.random.randint(0, len(test_images))]
    mask_name = image_name.replace(image_label, mask_label)

    input_image = cv2.imread(image_name)
    input_image = cv2.cvtColor(cv2.resize(input_image, (target_shape[0], target_shape[1]), cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)

    mask_img = cv2.imread(mask_name)
    mask_img = cv2.resize(mask_img, (target_shape[0], target_shape[1]), cv2.INTER_AREA)

    input_image = input_image / 255.0
    mask_img = mask_img / 255.0
    return input_image, mask_img


def display_images(images, masks, columns=5, width=15, height=17, max_images=15, plot=False):
    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images = images[0:max_images]
        masks = masks[0:max_images]
    i = 0
    final_image = []
    for image, mask in zip(images, masks):
        img = read_image_mask(image)

        label_mask = read_image_mask(mask, mask=True)
        label_mask = decode_segmentation_masks(label_mask)

        overlay = overlay_mask(img, label_mask)
        updated_img = cv2.hconcat([img, label_mask, overlay])
        if plot:
            height = max(height, int(len(images) / columns) * height)
            plt.figure(figsize=(height, width))
            plt.subplot(int(len(images) / columns + 1), columns, i + 1)
            plt.title("image-" + str(i + 1))
            plt.tight_layout()
            plt.imshow(updated_img)
        else:
            final_image.append(updated_img)
        i += 1
    return final_image


def display_predictions(mask, groud_truth, img):
    fig, axs = plt.subplots(1, 3, figsize=(8, 7))
    axs[2].imshow(mask[:, :, 0], cmap="gray")
    axs[2].set_title('Predicted')
    axs[1].imshow(groud_truth[:, :, 0], cmap="gray")
    axs[1].set_title('GroundTruth')
    axs[0].imshow(img)
    axs[0].set_title('Map Input Image')
    plt.show()
    return 0

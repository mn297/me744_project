import cv2
import numpy as np
import os


def pick_color(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load image including alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Error: Could not read image.")
        return

    print(f"Loaded image: {image_path}")
    print(f"Shape: {img.shape}")
    print("Click on the image to print RGBA values.")
    print("Press 'q' to quit.")

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get pixel value
            # OpenCV uses BGR order
            pixel = img[y, x]

            if len(pixel) == 4:
                b, g, r, a = pixel
                print(f"Pos: ({x}, {y}) | RGBA: [{r}, {g}, {b}, {a}]")
            elif len(pixel) == 3:
                b, g, r = pixel
                print(f"Pos: ({x}, {y}) | RGB: [{r}, {g}, {b}] (No Alpha)")

    cv2.namedWindow("Color Picker", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Color Picker", mouse_callback)

    # Resize window if image is too large
    height, width = img.shape[:2]
    if width > 1200 or height > 800:
        scale = min(1200 / width, 800 / height)
        cv2.resizeWindow("Color Picker", int(width * scale), int(height * scale))

    while True:
        cv2.imshow("Color Picker", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # img_path = "C:/Users/john/Documents/programming_dirty/me744_project/maskrcnn_coco/test_img/4997_rgb_0001_label.png"
    img_path = "C:/Users/john/Documents/programming_dirty/me744_project/maskrcnn_coco/test_img/3448_rgb_0001_withapples_label.png"

    # img_path = "C:/Users/john/Documents/programming_dirty/me744_project/datasets/image_envy_5000/0010_label_rgb_0001.png"

    pick_color(img_path)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector
import os
import numpy as np

# Path to the image
image_path = os.path.join(os.getcwd(), "captured_images", "rgb_image.png")

def line_select_callback(eclick, erelease):
    """
    Callback for line selection.
    """
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    w = abs(x2 - x1)
    h = abs(y2 - y1)
    x = min(x1, x2)
    y = min(y1, y2)
    
    print(f"\nSelected Region:")
    print(f"  x (left)  : {int(x)}")
    print(f"  y (top)   : {int(y)}")
    print(f"  w (width) : {int(w)}")
    print(f"  h (height): {int(h)}")
    print(f"  Format (x, y, w, h): {int(x)}, {int(y)}, {int(w)}, {int(h)}")
    print(f"  Format (xmin, ymin, xmax, ymax): {int(x)}, {int(y)}, {int(x+w)}, {int(y+h)}")

def toggle_selector(event):
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

if __name__ == '__main__':
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Please run 'python robotis_lab/scripts/save_camera_images.py ...' first.")
        exit(1)

    print(f"Loading image from: {image_path}")
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    print("\nInstructions:")
    print("1. Click and drag to draw a box around the region of interest (Robot + Cube).")
    print("2. The coordinates will be printed to this terminal.")
    print("3. Close the window when you are satisfied with the coordinates.")

    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    
    plt.connect('key_press_event', toggle_selector)
    plt.title("Draw a box around the ROI (Robot + Cube)")
    plt.show()

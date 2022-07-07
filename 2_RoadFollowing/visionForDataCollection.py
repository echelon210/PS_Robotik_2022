import cv2
import numpy as np

# FUNCITONALITIES:
# - process images to find centerlines
# - methods to get jetbot camera image


def drawLines(image, lines, color=[0, 0, 255], thickness=6):
    """ Method to draw lines into images. 
        In(1): image - The image to draw lines into
        In(2): lines - The lines to draw as np.array 
        In(3): color - The color the lines should have in BGR [default is Red]
        In(4): thickness - The thickness for the lines [default is 6] 
        Out(1): returns image with drawn lines
    """
    # If there are no lines to draw, exit.
    if lines is None:
        return

    # Make a copy of the original image and create blank image with same size
    image = np.copy(image)
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8,)

    # Loop over lines and draw them on the blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # Merge the image with the lines onto the original image and return it
    return cv2.addWeighted(image, 0.8, line_img, 1.0, 0.0)


def maskImage(image, heightPercent):
    """ Method to define a region of interest with masking an image. 
        In(1): image - The image that should be masked 
        In(2): heightPercent - The percent of the height that should be masked counted from bottom
        Out(1): returns a image where half of the image is masked with a black surface
    """
    height, width = image.shape
    mask = np.zeros_like(image)
    roiPercentageFromTop = 1 - heightPercent

    # A polygon defining the region that should be masked as np.array
    polygon = np.array([[
        (0, height * roiPercentageFromTop),  # Top left
        (width, height * roiPercentageFromTop),  # Top right
        (width, height),  # Bottom right
        (0, height),  # Bottom left
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)  # Overlay image with black polygon
    return cv2.bitwise_and(image, mask)


def findCenterline(image, show=False):
    """ Find and draw centerline in input image. 
        In(1): image - The image to process
        In(2): show - Boolean to indicate whether to show intermediate steps or not
        Out(1): image - image with centerline overlayed
        Out(2): foundCenterline - Determines whether centerline was found
        Out(3): angle - Angle to the centerline counted from middle of jetbot
    """
    angle = 0.0
    foundCenterline = None

    # Apply gaussian image blurring
    gaussKernelSize = (3, 3)
    blurred_img = cv2.GaussianBlur(image, gaussKernelSize, 0)

    # Convert to HSV image representation
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    # Filter out yellow of the center line
    low_yellow = np.array([5, 40, 100])
    up_yellow = np.array([50, 255, 255])
    col_img = cv2.inRange(hsv_img, low_yellow, up_yellow)

    # Filter out region of interest
    # Region of interest is lower quarter of image
    region_img = maskImage(col_img, 0.75)

    # Apply canny edge detection
    canny_img = cv2.Canny(region_img, 85, 150)

    lines = cv2.HoughLinesP(canny_img, rho=1, theta=np.pi/180,
                            threshold=30, lines=np.array([]), minLineLength=5, maxLineGap=50)
    if np.any(lines) == None:
        foundCenterline = False
    else:
        if show:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(blurred_img, (x1, y1), (x2, y2), (255, 0, 0),
                         thickness=4)  # Write lines into original

        # Center line detection
        lines_x = []
        lines_y = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                lines_x.extend([x1, x2])
                lines_y.extend([y1, y2])

        min_y = int(image.shape[0] * 0.25)
        max_y = image.shape[0]  # The bottom of the image

        poly = np.poly1d(np.polyfit(lines_y, lines_x, deg=1))
        # Start in the middle of the picture
        center_x_start = int(image.shape[1] * 0.5)
        center_x_end = int(poly(min_y))

        if (center_x_end - center_x_start) == 0:
            angle = 0
        else:
            angle = round(np.rad2deg(np.arctan2((max_y - min_y),
                          (center_x_start - center_x_end))) - 90, 3)

        foundCenterline = True
        if show:
            # Display line segments
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(blurred_img, (x1, y1), (x2, y2), (255, 0, 0),
                         thickness=4)  # Write lines into original
            # Display trajectory line
            image = drawLines(
                image, [[[center_x_start, max_y, center_x_end, min_y], ]], color=[0, 0, 255])

            # Display found centerline
            image = drawLines(
                image, [[[int(poly(max_y)), max_y, center_x_end, min_y], ]], color=[255, 0, 0])

            # Display angle in degrees
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(image, "Angle: " + str(angle), (25, 25),
                        font, fontScale, color, thickness, cv2.LINE_AA)

    if show:
        cv2.imshow("Centerline - Blurred image with detected lines", blurred_img)
        cv2.imshow("Centerline - Color filtered image", col_img)
        cv2.imshow("Centerline - Canny edge detection in ROI", canny_img)
        cv2.imshow("Centerline - Image with direction arrow", image)

    return image, foundCenterline, angle

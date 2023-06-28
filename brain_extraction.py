import os
import shutil
import cv2 as cv
import numpy as np

# Set threshold
THRESHOLD = 0.8
R_WIDTH = 4
R_HEIGHT = 5


def generateDir(dirName):
    """Create a directory if it doesn't exist or delete and recreate if it already exists.

    Args:
        dirName (str): Name of the directory to be created or recreated.
    """
    if os.path.exists(dirName):
        shutil.rmtree(dirName)
    os.mkdir(dirName)


def deleteBlankImages(imagePath):
    """Delete an image file if it is blank (contains only zeros).

    Args:
        imagePath (str): Path to the image file.

    Returns:
        bool: True if the image was blank and deleted, False otherwise.
    """
    image = cv.imread(imagePath, 0)
    if cv.countNonZero(image) == 0:
        os.remove(imagePath)
        return True
    return False


def templateMatching(grayscaleImage, template):
    """Perform template matching to find the best match between a grayscale image and a template.

    Args:
        grayscaleImage (numpy.ndarray): Grayscale image to be matched.
        template (numpy.ndarray): Template image.

    Returns:
        numpy.ndarray: Match result image.
    """
    return cv.matchTemplate(grayscaleImage, template, cv.TM_CCOEFF_NORMED)


def getDimensionOfImage(coordinates):
    """Calculate the width and height of an image based on the given pixel coordinates.

    Args:
        coordinates (list): List of pixel coordinates.

    Returns:
        int: Width of the image.
        int: Height of the image.
    """
    width = coordinates[1][1] - coordinates[0][1]
    height = 0
    last = -1
    for pixelCoord in coordinates:
        if last == -1:
            last = pixelCoord[0]
        else:
            if pixelCoord[0] != last:
                height = pixelCoord[0] - last
                last = pixelCoord[0]
                break

    return width, height


def cropImage(images, imagePath, slicedDirectory):
    """Crop the input image based on the location of a specific pattern and save the cropped slices.

    Args:
        images (str): Name of the image.
        imagePath (str): Path to the input image.
        slicedDirectory (str): Directory to save the sliced images.
    """
    coordinates = []
    imageRGB = cv.imread(imagePath)
    imageGrayscale = cv.cvtColor(imageRGB, cv.COLOR_BGR2GRAY)

    templateR = cv.imread('R.png', 0)
    res = templateMatching(imageGrayscale, templateR)

    loc = np.where(res >= THRESHOLD)

    for coord in zip(*loc[::-1]):
        coordinates.append((coord[1], coord[0]))

    width, height = getDimensionOfImage(coordinates)

    slicedImageDirectory = os.path.join(slicedDirectory, images.split(".")[0])
    generateDir(slicedImageDirectory)

    imageIndex = 1
    for pixelCoord in coordinates:
        diagonalCoord = (pixelCoord[0] + width, pixelCoord[1] + height)
        image = cv.imread(imagePath)
        croppedImage = image[pixelCoord[0] + R_HEIGHT:diagonalCoord[0] - R_HEIGHT,
                             pixelCoord[1] + R_WIDTH:diagonalCoord[1] - R_WIDTH]

        try:
            slicePath = os.path.join(slicedImageDirectory, str(imageIndex) + ".png")
            cv.imwrite(slicePath, croppedImage)
        except:
            break

        flag = deleteBlankImages(slicePath)
        if not flag:
            imageIndex += 1


def drawBoundaries(imageSlicePath, boundaryDirectoryPath, sliceName):
    """Draw contours around objects in an image and save the result.

    Args:
        imageSlicePath (str): Path to the image slice.
        boundaryDirectoryPath (str): Directory to save the boundary images.
        sliceName (str): Name of the slice image.
    """
    image = cv.imread(imageSlicePath)
    image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(image_grayscale, 10, 255, cv.THRESH_BINARY)
    contour, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contourImage = image.copy()
    cv.drawContours(contourImage, contour, -1, (255, 233, 0), 1, cv.LINE_AA)
    cv.imwrite(os.path.join(boundaryDirectoryPath, sliceName), contourImage)
    cv.destroyAllWindows()

import os
import brain_extraction
import shutil

from brain_extraction import generateDir

SLICE_DIR_PATH = "Slices"
BOUNDARY_DIR_PATH = "Boundaries"
IMAGE_DIR_PATH = "Data_1"

def main():
    """Main function to execute the brain extraction and contour drawing process."""
    
    # Create Slices Directory
    generateDir(SLICE_DIR_PATH)

    # Read data from testPatient and store image in Slices folder
    for images in os.listdir(IMAGE_DIR_PATH):
        if images.endswith("thresh.png"):
            imgPath = os.path.join(IMAGE_DIR_PATH, images)
            brain_extraction.cropImage(images, imgPath, SLICE_DIR_PATH)

    # Create Boundaries Directory
    generateDir(BOUNDARY_DIR_PATH)

    # Draw Contour and Store in Boundaries
    for sliceDirectory in os.listdir(SLICE_DIR_PATH):
        imageBoundaryDirectory = os.path.join(BOUNDARY_DIR_PATH, sliceDirectory)

        # Create nested folders for each image in Boundaries folder
        generateDir(imageBoundaryDirectory)
        for imageSlice in os.listdir(os.path.join(SLICE_DIR_PATH, sliceDirectory)):
            brain_extraction.drawBoundaries(os.path.join(SLICE_DIR_PATH, sliceDirectory, imageSlice), imageBoundaryDirectory, imageSlice)


if __name__ == "__main__":
    main()

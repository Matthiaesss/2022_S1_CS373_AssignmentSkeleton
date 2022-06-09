import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

# Convert RGB to greyscale
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            greyscale_pixel_array[i][j] = round(
                (0.299 * pixel_array_r[i][j]) + (0.587 * pixel_array_g[i][j]) + (0.114 * pixel_array_b[i][j]))

    return greyscale_pixel_array

# Compute Standard Deviation of pixel in neighbourhood 5x5
def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    greyScale = createInitializedGreyscalePixelArray(image_width, image_height)
    tempList = []
    mean = 0
    adding = 0
    for i in range(image_height):
        for j in range(image_width):
            if i == 0 or j == 0 or i == image_height - 1 or j == image_width - 1:
                continue
            else:
                if i - 2 >= 0 and j - 2 >= 0:
                    tempList.append(pixel_array[i - 2][j - 2])
                if i - 2 >= 0 and j - 1 >= 0:
                    tempList.append(pixel_array[i - 2][j - 1])
                if i - 2 >= 0:
                    tempList.append(pixel_array[i - 2][j])
                if i - 2 >= 0 and j + 1 <= image_width - 1:
                    tempList.append(pixel_array[i - 2][j + 1])
                if i - 2 >= 0 and j + 2 <= image_width - 1:
                    tempList.append(pixel_array[i - 2][j + 2])

                if i - 1 >= 0 and j - 2 >= 0:
                    tempList.append(pixel_array[i - 1][j - 2])
                if i - 1 >= 0 and j - 1 >= 0:
                    tempList.append(pixel_array[i - 1][j - 1])
                if i - 1 >= 0:
                    tempList.append(pixel_array[i - 1][j])
                if i - 1 >= 0 and j + 1 <= image_width - 1:
                    tempList.append(pixel_array[i - 1][j + 1])
                if i - 1 >= 0 and j + 2 <= image_width - 1:
                    tempList.append(pixel_array[i - 1][j + 2])

                if j - 2 >= 0:
                    tempList.append(pixel_array[i][j - 2])
                if j - 1 >= 0:
                    tempList.append(pixel_array[i][j - 1])
                tempList.append(pixel_array[i][j])
                if j + 1 <= image_width - 1:
                    tempList.append(pixel_array[i][j + 1])
                if j + 2 <= image_width - 1:
                    tempList.append(pixel_array[i][j + 2])

                if i + 1 <= image_height - 1 and j - 2 >= 0:
                    tempList.append(pixel_array[i + 1][j - 2])
                if i + 1 <= image_height - 1 and j - 1 >= 0:
                    tempList.append(pixel_array[i + 1][j - 1])
                if i + 1 <= image_height - 1:
                    tempList.append(pixel_array[i + 1][j])
                if i + 1 <= image_height - 1 and j + 1 <= image_width - 1:
                    tempList.append(pixel_array[i + 1][j + 1])
                if i + 1 <= image_height - 1 and j + 2 <= image_width - 1:
                    tempList.append(pixel_array[i + 1][j + 2])

                if i + 2 <= image_height - 1 and j - 2 >= 0:
                    tempList.append(pixel_array[i + 2][j - 2])
                if i + 2 <= image_height - 1 and j - 1 >= 0:
                    tempList.append(pixel_array[i + 2][j - 1])
                if i + 2 <= image_height - 1:
                    tempList.append(pixel_array[i + 2][j])
                if i + 2 <= image_height - 1 and j + 1 <= image_width - 1:
                    tempList.append(pixel_array[i + 2][j + 1])
                if i + 2 <= image_height - 1 and j + 2 <= image_width - 1:
                    tempList.append(pixel_array[i + 2][j + 2])

                for x in tempList:
                    mean += x

                mean = float(mean) / 25

                for y in tempList:
                    adding += (y - mean) ** 2

                adding = adding / len(tempList)
                greyScale[i][j] = math.sqrt(adding)
                mean = 0
                adding = 0
                tempList = []

    return greyScale

# Compute minimum and maximum values in pixel array
def computeMinAndMaxValues(pixel_array, image_width, image_height):
    minimum = pixel_array[0][0]
    maximum = pixel_array[0][0]

    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] < minimum:
                minimum = pixel_array[i][j]
            if pixel_array[i][j] > maximum:
                maximum = pixel_array[i][j]

    return (minimum, maximum)

# Contrast stretching
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    greyscalePixelArray = createInitializedGreyscalePixelArray(image_width, image_height)
    minAndMax = computeMinAndMaxValues(pixel_array, image_width, image_height)
    num = 0

    for i in range(image_height):
        for j in range(image_width):
            if minAndMax[1] - minAndMax[0] != 0:
                num = round((pixel_array[i][j] - minAndMax[0]) * (255 / (minAndMax[1] - minAndMax[0])))
            else:
                num = 0
            greyscalePixelArray[i][j] = num

    return greyscalePixelArray

# Compute threshold
def computeThresholdGE(pixel_array, threshold, image_width, image_height):
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] < threshold:
                pixel_array[i][j] = 0
            else:
                pixel_array[i][j] = 255

    return pixel_array

# Compute erosion 3x3
def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    greyScale = createInitializedGreyscalePixelArray(image_width, image_height)
    tempList = []
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            if pixel_array[i - 1][j - 1] > 0:
                if pixel_array[i - 1][j] > 0:
                    if pixel_array[i - 1][j + 1] > 0:
                        if pixel_array[i][j - 1] > 0:
                            if pixel_array[i][j] > 0:
                                if pixel_array[i][j + 1] > 0:
                                    if pixel_array[i + 1][j - 1] > 0:
                                        if pixel_array[i + 1][j] > 0:
                                            if pixel_array[i + 1][j + 1] > 0:
                                                greyScale[i][j] = 1
            else:
                greyScale[i][j] = 0

    return greyScale

# Compute dilation 3x3
def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    greyScale = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] > 0:
                for eta in [-1, 0, 1]:
                    for j1 in [-1, 0, 1]:
                        if (i + eta < image_height) and (i + eta >= 0):
                            if (j + j1 < image_width) and (j + j1 >= 0):
                                greyScale[i + eta][j + j1] = 1

    return greyScale

# Compute connected component labeling
def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    label = 1
    greyScale = createInitializedGreyscalePixelArray(image_width, image_height, 0)
    visited = createInitializedGreyscalePixelArray(image_width, image_height, 0)
    dictionary = {}
    for i in range(image_height):
        for j in range(image_width):
            if visited[i][j] == 0 and pixel_array[i][j] > 0:
                q = Queue()
                q.enqueue((i, j))
                dictionary[label] = 0
                visited[i][j] = 1
                while q.isEmpty() == False:
                    location = q.dequeue()
                    row = location[0]
                    col = location[1]
                    greyScale[row][col] = label
                    dictionary[label] += 1

                    left = pixel_array[row][col - 1]
                    right = pixel_array[row][col + 1]
                    top = pixel_array[row - 1][col]
                    bottom = pixel_array[row + 1][col]

                    if row < image_height and col - 1 < image_width and left > 0 and visited[row][col - 1] == 0:
                        q.enqueue((row, col - 1))
                        visited[row][col - 1] = 1
                        greyScale[row][col - 1] = label

                    if row < image_height and col + 1 < image_width and right > 0 and visited[row][col + 1] == 0:
                        q.enqueue((row, col + 1))
                        visited[row][col + 1] = 1
                        greyScale[row][col + 1] = label

                    if row - 1 < image_height and col < image_width and top > 0 and visited[row - 1][col] == 0:
                        q.enqueue((row - 1, col))
                        visited[row - 1][col] = 1
                        greyScale[row - 1][col] = label

                    if row + 1 < image_height and col < image_width and bottom > 0 and visited[row + 1][col] == 0:
                        q.enqueue((row + 1, col))
                        visited[row + 1][col] = 1
                        greyScale[row + 1][col] = label
                label += 1
            else:
                continue

    largest = 0
    group = 0
    topLeft = [0, 9999]
    botRight = [9999, 0]

    for key, value in dictionary.items():
        if value > largest:
            largest = value
            group = key

    for i in range(image_height):
        for j in range(image_width):
            if greyScale[i][j] == group:
                if i > topLeft[0]:
                    topLeft[0] = i
                if i < botRight[0]:
                    botRight[0] = i
                if j < topLeft[1]:
                    topLeft[1] = j
                if j > botRight[1]:
                    botRight[1] = j

    return [topLeft, botRight]


# Combine red, green, and blue arrays into one pixel array
def computeRGBImage(red, green, blue, image_width, image_height):
    rgb = []
    for i in range(image_height):
        row = []
        for j in range(image_width):
            all = []
            all.append(red[i][j])
            all.append(green[i][j])
            all.append(blue[i][j])
            row.append(all)
        rgb.append(row)
    return rgb


# Queue class for computeConnectedComponentLabeling() function
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate5.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here

    # RGB to greyscale
    px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    # High contrast standard deviation
    px_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)

    # Contrast stretching
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)

    # High contrast binary image
    px_array = computeThresholdGE(px_array, 143, image_width, image_height)

    # Dilate image
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)

    # Errode image
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)

    # Connected component labeling
    box = computeConnectedComponentLabeling(px_array, image_width, image_height)

    # Combine red, green, and blue pixel arrays to output
    px_array = computeRGBImage(px_array_r, px_array_g, px_array_b, image_width, image_height)

    # Compute the bounding box for license plate
    bbox_min_x = box[0][1]
    bbox_max_x = box[1][1]
    bbox_min_y = box[1][0]
    bbox_max_y = box[0][0]



    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)



    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()
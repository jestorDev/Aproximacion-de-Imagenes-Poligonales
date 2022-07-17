from deap import base
from deap import creator
from deap import tools

import random
import numpy  as np
import os

import image_test
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
from deap import algorithms

from operator import attrgetter


# problem related constants
POLYGON_SIZE = 3
NUM_OF_POLYGONS = 100

# calculate total number of params in chromosome:
# For each polygon we have:
# two coordinates per vertex, 3 color values, one alpha value
NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)

# Genetic Algorithm constants:
POPULATION_SIZE = 200
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # probability for mutating an individual
MAX_GENERATIONS = 100
#HALL_OF_FAME_SIZE = 20
#CROWDING_FACTOR = 10.0  # crowding factor for crossover and mutation

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the image test class instance:
imageTest = image_test.ImageTest("images/logodna50.png", POLYGON_SIZE)

# calculate total number of params in chromosome:
# For each polygon we have:
# two coordinates per vertex, 3 color values, one alpha value
NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)

# all parameter values are bound between 0 and 1, later to be expanded:
BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0  # boundaries for all dimensions

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)

# helper function for creating random real numbers uniformly distributed within a given range [low, up]
# it assumes that the range is the same for every dimension
def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * NUM_OF_PARAMS, [up] * NUM_OF_PARAMS)]

# create an operator that randomly returns a float in the desired range:
toolbox.register("attrFloat", randomFloat, BOUNDS_LOW, BOUNDS_HIGH)

# create an operator that fills up an Individual instance:
toolbox.register("individualCreator",
                 tools.initIterate,
                 creator.Individual,
                 toolbox.attrFloat)

# create an operator that generates a list of individuals:
toolbox.register("populationCreator",
                 tools.initRepeat,
                 list,
                 toolbox.individualCreator)

##########################################################################################
    #Fitnesss


class ImageTest:

    def __init__(self, imagePath, polygonSize):
        """
        Initializes an instance of the class
        :param imagePath: the path of the file containing the reference image
        :param polygonSize: the number of vertices on the polygons used to recreate the image
        """
        self.refImage = Image.open(imagePath)
        self.polygonSize = polygonSize

        self.width, self.height = self.refImage.size
        self.numPixels = self.width * self.height
        self.refImageCv2 = self.toCv2(self.refImage)


    def polygonDataToImagea(self, polygonData):
        
        """
        accepts polygon data and creates an image containing these polygons.
        :param polygonData: a list of polygon parameters. Each item in the list
        represents the vertices locations, color and transparency of the corresponding polygon
        :return: the image containing the polygons (Pillow format)
        """
        #print (len(polygonData))

        #print ("---------------Polygon data------------------------------")
        #print (polygonData)



        
        # start with a new image:

        ima = np.ones((self.width,self.height,3), np.int8)
        #image = Image.new('RGB', (self.width, self.height))#TODO
        #draw = ImageDraw.Draw(image, 'RGBA')

        # divide the polygonData to chunks, each containing the data for a single polygon:
        chunkSize = self.polygonSize * 2 + 4  # (x,y) per vertex + (RGBA)
        
        arrR = np.array(polygonData).reshape(-1 , chunkSize)
        arrVertex = arrR[ :,:self.polygonSize*2 ].reshape( (-1 ,self.polygonSize,2)) 
        arrColor = arrR[ :,self.polygonSize*2:-1 ]

        arrVertex[ : , : , 1]  *= self.height
        arrVertex[ : , : , 0]  *= self.width 
        arrColor *= 255

        for vertexArr , arrColor in zip(arrVertex , arrColor):
            #print(vertexArr , arrColor)
            
            cv2.drawContours(ima,  np.int32([vertexArr]), 0, tuple(arrColor), -1)
            #                                                      r   g   b
            # cv2.fillPoly(ima, pts = np.int32([vertexArr]) , color=(0, 0, 0 ))
            #draw.polygon(vertices, (red, green, blue, alpha))

        # cleanup:
        #del draw
        # https://gist.github.com/IAmSuyogJadhav/305bfd9a0605a4c096383408bee7fd5c


        return   cv2.cvtColor(ima, cv2.COLOR_RGB2BGR)

    def polygonDataToImage(self, polygonData):
        """
        accepts polygon data and creates an image containing these polygons.
        :param polygonData: a list of polygon parameters. Each item in the list
        represents the vertices locations, color and transparency of the corresponding polygon
        :return: the image containing the polygons (Pillow format)
        """

        # start with a new image:
        image = Image.new('RGB', (self.width, self.height))#TODO
        draw = ImageDraw.Draw(image, 'RGBA')

        # divide the polygonData to chunks, each containing the data for a single polygon:
        chunkSize = self.polygonSize * 2 + 4  # (x,y) per vertex + (RGBA)
        polygons = self.list2Chunks(polygonData, chunkSize)

        # iterate over all polygons and draw each of them into the image:
        for poly in polygons:
            index = 0

            # extract the vertices of the current polygon:
            vertices = []
            for vertex in range(self.polygonSize):
                vertices.append((int(poly[index] * self.width), int(poly[index + 1] * self.height)))
                index += 2

            # extract the RGB and alpha values of the current polygon:
            red = int(poly[index] * 255)
            green = int(poly[index + 1] * 255)
            blue = int(poly[index + 2] * 255)
            alpha = int(poly[index + 3] * 255)

            # draw the polygon into the image:
            draw.polygon(vertices, (red, green, blue, alpha))

        # cleanup:
        del draw

        return image

    def getDifference(self, polygonData, method="MSE"):
        """
        accepts polygon data, creates an image containing these polygons, and calculates the difference
        between this image and the reference image using one of two methods.
        :param polygonData: a list of polygon parameters. Each item in the list
        represents the vertices locations, color and transparency of the corresponding polygon
        :param method: base method of calculating the difference ("MSE" or "SSIM").
        larger return value always means larger difference
        :return: the calculated difference between the image containg the polygons and the reference image
        """

        # create the image containing the polygons:
        image = self.polygonDataToImage(polygonData)

        if method == "MSE":
            return self.getMse(image)
        else:
            return 1.0 - self.getSsim(image)

    def plotImages(self, image, header=None):
        """
        creates a 'side-by-side' plot of the given image next to the reference image
        :param image: image to be drawn next to reference image (Pillow format)
        :param header: text used as a header for the plot
        """

        fig = plt.figure("Image Comparison:")
        if header:
            plt.suptitle(header)

        # plot the reference image on the left:
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(self.refImage)
        self.ticksOff(plt)

        # plot the given image on the right:
        fig.add_subplot(1, 2, 2)
        plt.imshow(image)
        self.ticksOff(plt)

        return plt

    def saveImage(self, polygonData, imageFilePath, header=None):
        """
        accepts polygon data, creates an image containing these polygons,
        creates a 'side-by-side' plot of this image next to the reference image,
        and saves the plot to a file
        :param polygonData: a list of polygon parameters. Each item in the list
        represents the vertices locations, color and transparency of the corresponding polygon
        :param imageFilePath: path of file to be used to save the plot to
        :param header: text used as a header for the plot
        """

        # create an image from th epolygon data:
        image = self.polygonDataToImage(polygonData)

        # plot the image side-by-side with the reference image:
        self.plotImages(image, header)

        # save the plot to file:
        plt.savefig(imageFilePath)

    # utility methods:

    def toCv2(self, pil_image):
        """converts the given Pillow image to CV2 format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def getMse(self, image):
        """calculates MSE of difference between the given image and the reference image"""
        return np.sum((self.toCv2(image).astype("float") - self.refImageCv2.astype("float")) ** 2)/float(self.numPixels)

    def getMseCV(self, image):
        """calculates MSE of difference between the given image and the reference image"""
        return np.sum((image.astype("float") - self.refImageCv2.astype("float")) ** 2)/float(self.numPixels)


    def getSsim(self, image):
        """calculates mean structural similarity index between the given image and the reference image"""
        return structural_similarity(self.toCv2(image), self.refImageCv2, multichannel=True)

    def list2Chunks(self, list, chunkSize):
        """divides a given list to fixed size chunks, returns a generator iterator"""
        for chunk in range(0, len(list), chunkSize):
            yield(list[chunk:chunk + chunkSize])

    def ticksOff(self, plot):#TODO
        """turns off ticks on both axes"""
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            top=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )




# fitness calculation using MSE as difference metric:
def getDiff(individual):
    return imageTest.getDifference(individual, "MSE"),
    #return imageTest.getDifference(individual, "SSIM"),

toolbox.register("evaluate", getDiff)

##########################################################################################


# genetic operators:
toolbox.register("select", selccionTorneo , tournsize=2)

toolbox.register("mate",
                 tools.cxTwoPoint)




def mutUniformFloat(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """

    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.uniform(low, up)

    return individual,

toolbox.register("mutate",
                 mutUniformFloat,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 indpb=1.0/NUM_OF_PARAMS)


# save the best current drawing every 100 generations (used as a callback):
def saveImage(gen, polygonData):

    # only every 100 generations:
    if gen % 100 == 0:

        # create folder if does not exist:
        folder = "images/results/run-{}-{}".format(POLYGON_SIZE, NUM_OF_POLYGONS)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # save the image in the folder:
        imageTest.saveImage(polygonData,
                            "{}/after-{}-gen.png".format(folder, gen),
                            "After {} Generations".format(gen))

# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    #hof = tools.HallOfFame(HALL_OF_FAME_SIZE)


    # perform the Genetic Algorithm flow with elitism and 'saveImage' callback:
    """
    population, logbook = elitism_callback.eaSimpleWithElitismAndCallback(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      callback=saveImage,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)
    """
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
                                   stats=stats, verbose=True)


    # print best solution found:
    best = population[0]
    print()
    print("Best Solution = ", best)
    print("Best Score = ", best.fitness.values[0])
    print()
    # draw best image next to reference image:
    imageTest.plotImages(imageTest.polygonDataToImage(best))

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    print("Fitness minimo : ")
    print(minFitnessValues)
    
    print("Fitness promedio : ")
    print(meanFitnessValues)
    # plot statistics:
    sns.set_style("whitegrid")
    plt.figure("Stats:")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # show both plots:
    plt.show()

if __name__ == "__main__":
    main()

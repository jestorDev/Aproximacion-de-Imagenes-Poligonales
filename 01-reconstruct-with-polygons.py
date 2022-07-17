from deap import base
from deap import creator
from deap import tools

import random
import numpy  as np
import os

#import image_test
import cv2
from PIL import Image, ImageDraw

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
MAX_GENERATIONS = 3
#HALL_OF_FAME_SIZE = 20
#CROWDING_FACTOR = 10.0  # crowding factor for crossover and mutation

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

class ImageTest:

    def __init__(self, imagePath, polygonSize):
        self.refImage = Image.open(imagePath)
        self.polygonSize = polygonSize

        self.width, self.height = self.refImage.size
        self.numPixels = self.width * self.height
        self.refImageCv2 = self.toCv2(self.refImage)


    def polygonDataToImageb(self, polygonData):
        
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

        image = Image.new('RGB', (self.width, self.height))#TODO
        draw = ImageDraw.Draw(image, 'RGBA')

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
            
            #cv2.drawContours(ima,  np.int32([vertexArr]), 0, tuple(arrColor), -1)
            draw.polygon(np.int32(vertexArr),tuple(np.int32(arrColor)))
            #                                                      r   g   b
            # cv2.fillPoly(ima, pts = np.int32([vertexArr]) , color=(0, 0, 0 ))
            #draw.polygon(vertices, (red, green, blue, alpha))

        # cleanup:
        #del draw
        # https://gist.github.com/IAmSuyogJadhav/305bfd9a0605a4c096383408bee7fd5c

        del draw

        return image

    def polygonDataToImage(self, polygonData):
        """
        Toma una lista de poligonos  de la forma [x1,y1,x1,y1,x1,y1,r,g,b,a]
        y renderiza una imagen
        """

        image = Image.new('RGB', (self.width, self.height))
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
        # renderizado de imagen
        image = self.polygonDataToImage(polygonData)
        return self.getMse(image)

    def plotImages(self, image, header=None):
        fig = plt.figure("Comparacion imagen referecia vs generada:")
        if header:
            plt.suptitle(header)

        # grafica izquierda
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(self.refImage)
        self.ticksOff(plt)

        # grfica derecha
        fig.add_subplot(1, 2, 2)
        plt.imshow(image)
        self.ticksOff(plt)

        return plt

    def saveImage(self, polygonData, imageFilePath, header=None):
        image = self.polygonDataToImage(polygonData)
        self.plotImages(image, header)
        plt.savefig(imageFilePath)

    # utility methods:

    def toCv2(self, pil_image):
        """converts the given Pillow image to CV2 format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def getMse(self, image):
        """Calculo del error cuadratico medio"""
        return np.sum((self.toCv2(image).astype("float") - self.refImageCv2.astype("float")) ** 2)/float(self.numPixels)

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



# create the image test class instance:
imageTest = ImageTest("images/logodna50.png", POLYGON_SIZE)

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





# fitness calculation using MSE as difference metric:
def getDiff(individual):
    return imageTest.getDifference(individual),

toolbox.register("evaluate", getDiff)

##########################################################################################
##########################################################################################
# Metodo de seleccion de participantes
##########################################################################################


def selccionTorneo(individuals, k, tournsize, fit_attr="fitness"):
    chosen = []
    for i in range(k):
        aspirants = [random.choice(individuals) for i in range(tournsize)]
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen


# genetic operators:
toolbox.register("select", selccionTorneo , tournsize=2)

def crossoverDosPuntos(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    return ind1, ind2


def crossoverUnPunto(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2

toolbox.register("mate",
                 crossoverUnPunto)


def mutUniforme(individual, low, up, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.uniform(low, up)

    return individual,

toolbox.register("mutate",
                 mutUniforme,
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
                            "{}/aftercx1p-{}-gen.png".format(folder, gen),
                            "After {} Generations".format(gen))

# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)


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

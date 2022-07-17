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

    def __init__(self, imagenPath, nVertices):
        self.refImage = Image.open(imagenPath)
        self.nVertices = nVertices

        self.width, self.height = self.refImage.size
        self.refImageCv2 = self.toCv2(self.refImage)


    def poligonoAImagen(self, poligonos):
        """
        Toma una lista de poligonos  de la forma [x1,y1,x1,y1,x1,y1,r,g,b,a]
        y renderiza una imagen
        """

        img = Image.new('RGB', (self.width, self.height))
        imgDraw = ImageDraw.Draw(img, 'RGBA')

        lenPoligono = self.nVertices * 2 + 4  # (x,y) * numVertices + (RGBA)
        
        polygons = [ poligonos[chunk:chunk + lenPoligono] for chunk in range(0, len(poligonos), lenPoligono)  ]
        
        for poly in polygons:
            # vertices del poligono actual
            vertices = []
            index = 0
            for _ in range(self.nVertices):
                vertices.append((int(poly[index] * self.width), int(poly[index + 1] * self.height)))
                index += 2

            red = int(poly[index] * 255)
            green = int(poly[index + 1] * 255)
            blue = int(poly[index + 2] * 255)
            alpha = int(poly[index + 3] * 255)

            # renderizar un poligono
            imgDraw.polygon(vertices, (red, green, blue, alpha))

        del imgDraw
        return img

    def diferenciaImagenes(self, poligonos):
        # renderizado de imagen
        image = self.poligonoAImagen(poligonos)
        return self.getMse(image)

    def mostrarPlot(self, image, header=None):
        fig = plt.figure("Comparacion imagen referecia vs generada:")
        if header:
            plt.suptitle(header)

        # grafica izquierda
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(self.refImage)
        
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


        # grfica derecha
        fig.add_subplot(1, 2, 2)
        plt.imshow(image)
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


        return plt

    def save(self, poligonos, imageFilePath, header=None):
        image = self.poligonoAImagen(poligonos)
        self.mostrarPlot(image, header)
        plt.savefig(imageFilePath)

    # utility methods:

    def toCv2(self, pil_image):
        #Conversion de Pillow a  > Cv2 image
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def getMse(self, image):
        #Calculo del error cuadratico medio
        return np.sum((self.toCv2(image).astype("float") - self.refImageCv2.astype("float")) ** 2)/float(self.width * self.height)



imageTest = ImageTest("images/logodna50.png", POLYGON_SIZE)

NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)

MINV, MAXV = 0.0, 1.0  

toolbox = base.Toolbox()

# Objetivo minimizar el fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Individus seran listas 
creator.create("Individual", list, fitness=creator.FitnessMin)


def randomFloat(low, up):
    # Genera un poligono random de la forma 
    # Toma una lista de poligonos  de la forma [x1,y1,x1,y1,x1,y1,r,g,b,a]
    return [random.uniform(l, u) for l, u in zip([low] * NUM_OF_PARAMS, [up] * NUM_OF_PARAMS)]


#randomFloat se usara para crear los individuos
toolbox.register("attrFloat", randomFloat, MINV, MAXV)

toolbox.register("individualCreator",
                 tools.initIterate,
                 creator.Individual,
                 toolbox.attrFloat)

toolbox.register("populationCreator",
                 tools.initRepeat,
                 list,
                 toolbox.individualCreator)

##########################################################################################
#Fitnesss
##########################################################################################

# fitness calculation using MSE as difference metric:
def getDiff(individual):
    return imageTest.diferenciaImagenes(individual),

toolbox.register("evaluate", getDiff)

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

##########################################################################################
# Crossover
##########################################################################################


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

                 
##########################################################################################
# Mutacion
##########################################################################################


def mutUniforme(individual, low, up, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.uniform(low, up)

    return individual,

toolbox.register("mutate",
                 mutUniforme,
                 low=MINV,
                 up=MAXV,
                 indpb=1.0/NUM_OF_PARAMS)


def guardar(gen, poligonos):

    # cada 100 gens
    if gen % 100 == 0:

        folder = "images/resultado/exp-{}-{}".format(POLYGON_SIZE, NUM_OF_POLYGONS)
        if not os.path.exists(folder):
            os.makedirs(folder)

        imageTest.save(poligonos,
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
    imageTest.mostrarPlot(imageTest.poligonoAImagen(best))

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

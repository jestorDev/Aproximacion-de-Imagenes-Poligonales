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

from operator import attrgetter

import sys


SHOW = False
# problem related constants
NVERTICESPOL = 3
NUM_POLIGONOS = 100

# [x1,y1,x2,y2,x3,y3,r,g,b,a]
#     2     2    2      4
N_PARAMS = NUM_POLIGONOS * (NVERTICESPOL * 2 + 4)

params = sys.argv[1].split(",")
POP_SIZE = int(params[0])
P_CRUZA = float(params[1])
P_MUTACION = float(params[2])
MAX_GENS = 3

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
        Toma una lista de poligonos  de la forma  [x1,y1,x2,y2,x3,y3,r,g,b,a]
        y renderiza una imagen
        """

        img = Image.new('RGB', (self.width, self.height))
        imgDraw = ImageDraw.Draw(img, 'RGBA')

        lenPoligono = self.nVertices * 2 + 4  # (x,y) * numVertices + (RGBA)
        
        polygons = [ poligonos[idxPol:idxPol + lenPoligono] for idxPol in range(0, len(poligonos), lenPoligono)  ]
        
        for poly in polygons:
            # vertices del poligono actual
            vertices = []
            idx = 0
            for _ in range(self.nVertices):
                vertices.append((int(poly[idx] * self.width), int(poly[idx + 1] * self.height)))
                idx += 2

            red = int(poly[idx] * 255)
            green = int(poly[idx + 1] * 255)
            blue = int(poly[idx + 2] * 255)
            alpha = int(poly[idx + 3] * 255)

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



imageTest = ImageTest("images/logodna50.png", NVERTICESPOL)

N_PARAMS = NUM_POLIGONOS * (NVERTICESPOL * 2 + 4)

MINV, MAXV = 0.0, 1.0  

toolbox = base.Toolbox()

# Objetivo minimizar el fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Individus seran listas 
creator.create("Individual", list, fitness=creator.FitnessMin)


def randomFloat(low, up):
    # Genera un poligono random de la forma 
    # Toma una lista de poligonos  de la forma [x1,y1,x2,y2,x3,y3,r,g,b,a]
    return [random.uniform(l, u) for l, u in zip([low] * N_PARAMS, [up] * N_PARAMS)]


#randomFloat se usara para crear los individuos
toolbox.register("attrFloat", randomFloat, MINV, MAXV)

toolbox.register("individualCreator",
                 tools.initIterate,
                 creator.Individual,
                 toolbox.attrFloat)

toolbox.register("poblacionCreator",
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
                 indpb=1.0/N_PARAMS)


def guardar(gen, poligonos):

    # cada 100 gens
    if gen % 100 == 0:

        folder = "images/resultado/exp-{}-{}".format(NVERTICESPOL, NUM_POLIGONOS)
        if not os.path.exists(folder):
            os.makedirs(folder)

        imageTest.save(poligonos,
                            "{}/aftercx1p-{}-gen.png".format(folder, gen),
                            "After {} Generations".format(gen))





def cruzarMutar(poblacion, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in poblacion]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def algoGenetico(poblacion, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in poblacion if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(poblacion)

    record = stats.compile(poblacion) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(poblacion, len(poblacion))

        # Vary the pool of individuals
        offspring = cruzarMutar(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current poblacion by the offspring
        poblacion[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(poblacion) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return poblacion, logbook



# Genetic Algorithm flow:
def main():

    # create initial poblacion (generation 0):
    poblacion = toolbox.poblacionCreator(n=POP_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)


    poblacion, logbook =  algoGenetico(poblacion, toolbox, cxpb=P_CRUZA, mutpb=P_MUTACION, ngen=MAX_GENS,
                                   stats=stats, verbose=True)


    # print best solution found:

    print ("Pop Size", " PCruza" ,  " P_MUTACION")
    print (POP_SIZE , " " , P_CRUZA ," " , P_MUTACION)
    
    
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    print("Fitness minimo : ")
    print(minFitnessValues)
    print("Fitness promedio : ")
    print(meanFitnessValues)

    if SHOW :
        best = poblacion[0]
        print("Best Solution = ", best)
        print("Best Score = ", best.fitness.values[0])


        # draw best image next to reference image:
        imageTest.mostrarPlot(imageTest.poligonoAImagen(best))

        # extract statistics:


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

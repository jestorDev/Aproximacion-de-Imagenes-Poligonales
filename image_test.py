from PIL import Image, ImageDraw
import numpy as np
from skimage.metrics import structural_similarity
import cv2
import matplotlib.pyplot as plt

MAX_STEPS = 200
FLAG_LOCATION = 0.5

import numpy as np
from testing import getXYZtoUVWT, getTestData, getBraggForTrack
import matplotlib.pyplot as plt

def vLength(v :np.ndarray):
    return np.sqrt(np.sum(np.square(v)))

def getDensity(points, histograms):
    result = []
    for point in points.astype(np.int32):
        result += [np.array([histograms[i][point[3]][point[i]] for i in range(3)])]
    return np.array([np.sum(res) / 3 for res in result])

def estimateMiddle(images, tanget, vertex, start, end):
    direction_vector = (end - start) / vLength(end - start) # vector going alongside the line of length one mm
    direction_vector = direction_vector / 4 # now we have 0.25 mm which is the resolution of our distributions
    
    point = start.copy()
    measuring_points = []
    while vLength(point - start) < vLength(end - start):
        measuring_points += [point.copy()]
        point += direction_vector
    measuring_points = np.array(measuring_points)
    measuring_points_uvwt = getXYZtoUVWT(measuring_points)
    density = getDensity(measuring_points_uvwt, images)
    
    last_best_point = None
    last_best_score = np.inf
    for point in measuring_points:
        bragg = getBraggForTrack(np.array([start, point, end]), (end + start) / 2)[2]
        bragg = bragg[bragg[:,0] <= 75.0]
        score = vLength(density - bragg[:,1])
        
        if score < last_best_score:
            last_best_point = point
            last_best_score = score
        
    return last_best_point
estimateMiddle(*getTestData('middle'))
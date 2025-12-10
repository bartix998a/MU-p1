import numpy as np
from testing import getXYZtoUVWT, getTestData, getBraggForTrack
import matplotlib.pyplot as plt

from itertools import product

def vLength(v :np.ndarray):
    return np.sqrt(np.sum(np.square(v)))

def vPerp(v:np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])

# gets projections of the points that lie closer than epsilon to the line
def getProjections(points, vals, p0, d, epsilon):
    v = points.astype(np.float32) - p0
    
    proj = (np.dot(v, d)[:, None] * d) + p0
    
    return proj[np.linalg.norm(v - (proj - p0), axis=1) < epsilon], vals[np.linalg.norm(v - (proj - p0), axis=1) < epsilon]

# gets the distribution on a line from a single image through the projection method
def getDistributionProjections(points, epsilon, histogram):
    direction = points[-1,:] - points[0,:]
    
    direction = direction / vLength(direction)
        
    histogram_coord = np.array([np.array(tup) for tup in product(list(range(histogram.shape[0])), list(range(histogram.shape[1]))) if histogram[*tup] != 0])
    
    vals = histogram[histogram_coord[:,0], histogram_coord[:,1]]
    
    direction_perp = vPerp(direction)
    
    distribution = []
    
    for point in points:
        histogram_coord_p, vals_p = getProjections(histogram_coord[:,::-1], vals, point, direction_perp, epsilon)
        
        distribution += [np.average(vals_p, weights = 1 / (np.linalg.norm(histogram_coord_p - point, axis=1) + 1), axis=0) if vals_p.shape[0] != 0 else 0.0]
    
    return np.array(distribution)

def getDensity(points, histograms):
    result = []
    for point in points.astype(np.int32):
        result += [np.array([histograms[point[3]][point[i]][i] for i in range(3)])]
    return np.array([np.sum(res) / 3 for res in result])

def estimateMiddle(images, start, end):
    direction_vector = (end - start) / vLength(end - start) # vector going alongside the line of length one mm
    direction_vector = direction_vector / 4 # now we have 0.25 mm which is the resolution of our distributions
    
    point = start.copy()
    measuring_points = []
    while vLength(point - start) < vLength(end - start):
        measuring_points += [point.copy()]
        point += direction_vector
    measuring_points = np.array(measuring_points)
    measuring_points_uvwt = getXYZtoUVWT(measuring_points)
 
    distribution = np.average(
        np.array([getDistributionProjections(measuring_points_uvwt[:,[i,3]], 4.0, images[i]) for i in range(3)]), 
        axis=0)
    
    last_best_point = None
    last_best_score = np.inf
    for point in measuring_points:
        bragg = getBraggForTrack(np.array([start, point, end]), (end + start) / 2)[2]
        bragg = bragg[bragg[:,0] <= vLength(end - start)]
        score = vLength(distribution - bragg[:,1])
        
        if score < last_best_score:
            last_best_point = point
            last_best_score = score
    
    return last_best_point
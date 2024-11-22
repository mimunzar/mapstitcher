import numpy as np
from scipy.optimize import minimize

def reprojection_error(h, p, q):
        h = h.reshape((3, 3))
        p_proj = np.dot(h, np.append(p, 1))
        p_proj /= p_proj[2]  # Normalize by homogeneous coordinate
        return np.linalg.norm(p_proj[:2] - q)

def objective(h_flat, correspondences):
    h_matrices = [h_flat[i:i+9].reshape(3, 3) for i in range(0, len(h_flat), 9)]
    total_error = 0
    for (p, q, h_idx) in correspondences:
        total_error += reprojection_error(h_matrices[h_idx], p, q)
    return total_error

def homography_constraints(h_flat):
    h_matrices = [h_flat[i:i+9].reshape(3, 3) for i in range(0, len(h_flat), 9)]
    return [np.linalg.det(H) for H in h_matrices]  # One constraint per homography

class HomographyOptimizer:
    def __init__(self, max_matches=200, silent=False):
        self.max_matches = max_matches
        self.silent = silent

    def optimize(self, pairs, homographies, correspondences):
        if not self.silent:
            print('Optimizing homography...')
        # flatten homographies
        h_flat = np.array([h.flatten() for h in homographies]).flatten()

        # flatten correspondences
        correspondences_flat = []
        for homography_index, (points1, points2) in enumerate(correspondences):
            for p, q in zip(points1, points2):
                correspondences_flat.append((p, q, homography_index))

        # subsample correspondences
        #print('Number of correspondences:', len(correspondences_flat))
        subsample_factor = (int)(max(1, len(correspondences_flat) / (self.max_matches * len(homographies))))
        correspondences_flat = correspondences_flat[::subsample_factor]

        # print size of correspondences
        #print('Number of correspondences:', len(correspondences_flat))
        if not self.silent:
            print('Initial error:', objective(h_flat, correspondences_flat))

        constraints = [{'type': 'ineq', 'fun': lambda h_flat, idx=i: np.linalg.det(h_flat[idx*9:(idx+1)*9].reshape(3, 3))}
               for i in range(len(homographies))]
        # todo: find solver that supports constraints
        
        result = minimize(
            objective,
            h_flat,
            args=(correspondences_flat,),
            method='trust-constr',
            #method='BFGS',
            options={'disp': not self.silent}
        )
        result = minimize(objective, h_flat, args=(correspondences_flat,))
        if not self.silent:
            print('Final error:', objective(result.x, correspondences_flat))

        optimized_h = [result.x[i:i+9].reshape(3, 3) for i in range(0, len(result.x), 9)]

        return optimized_h

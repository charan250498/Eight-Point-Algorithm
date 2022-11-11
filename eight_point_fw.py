import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

def find_matching_keypoints(image1, image2):
    #Input: two images (numpy arrays)
    #Output: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    #img1: image on which we draw the epilines for the points in img2
    #lines: corresponding epilines
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def get_transformation_matrix(num_points, pts_mean, pts_distance_to_mean):
    transformation_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        transformation_matrix[i, i] = (2 ** 0.5)/pts_distance_to_mean
    transformation_matrix_x = np.hstack((transformation_matrix, np.array([1 - ((2 ** 0.5) * pts_mean[0])/pts_distance_to_mean] * num_points)[:, None]))
    transformation_matrix_y = np.hstack((transformation_matrix, np.array([1 - ((2 ** 0.5) * pts_mean[1])/pts_distance_to_mean] * num_points)[:, None]))
    return transformation_matrix_x, transformation_matrix_y

def normalize_points(pts):
    pts_mean = np.mean(pts, axis = 0)
    pts_distance_to_mean = np.mean(np.sum((pts - pts_mean) ** 2, axis = 1) ** 0.5)
    transformation_matrix_x, transformation_matrix_y = get_transformation_matrix(pts.shape[0], pts_mean, pts_distance_to_mean)
    pts = np.vstack((pts, np.array([1, 1])))
    pts_x = transformation_matrix_x @ pts[:, 0][:, None]
    pts_y = transformation_matrix_y @ pts[:, 1][:, None]
    return np.hstack((pts_x, pts_y))

def FindFundamentalMatrix(pts1, pts2):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    #todo: Normalize the points
    pts1 = normalize_points(pts1)
    pts2 = normalize_points(pts2)

    #todo: Form the matrix A
    A = np.zeros((8, 8))
    
    A[:, 0] = pts1[:8, 0] * pts2[:8, 0]
    A[:, 1] = pts1[:8, 0] * pts2[:8, 1]
    A[:, 2] = pts1[:8, 0]
    A[:, 3] = pts1[:8, 1] * pts2[:8, 0]
    A[:, 4] = pts1[:8, 1] * pts2[:8, 1]
    A[:, 5] = pts1[:8, 1]
    A[:, 6] = pts2[:8, 0]
    A[:, 7] = pts2[:8, 1]
    
    A = np.hstack((A, np.ones(8)[:, None]))

    #todo: Find the fundamental matrix
    u, sigma, v = np.linalg.svd(A)
    fundamental_matrix = v[-1, :].reshape(3,3)

    return fundamental_matrix

def FindFundamentalMatrixRansac(pts1, pts2, num_trials = 1000, threshold = 0.01):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    #todo: Run RANSAC and find the best fundamental matrix
    raise NotImplementedError

if __name__ == '__main__':
    #Set parameters
    data_path = './data'
    use_ransac = False

    #Load images
    image1_path = os.path.join(data_path, 'myleft.jpg')
    image2_path = os.path.join(data_path, 'myright.jpg')
    image1 = np.array(Image.open(image1_path).convert('L'))
    image2 = np.array(Image.open(image2_path).convert('L'))


    #Find matching keypoints
    pts1, pts2 = find_matching_keypoints(image1, image2)

    #Builtin opencv function for comparison
    F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]

    #todo: FindFundamentalMatrix
    if use_ransac:
        F = FindFundamentalMatrixRansac(pts1, pts2)
    else:
        F = FindFundamentalMatrix(pts1, pts2)

    # Find epilines corresponding to points in second image,  and draw the lines on first image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F_true)
    lines1 = lines1.reshape(-1, 3)
    img1, img2 = drawlines(image1, image2, lines1, pts1, pts2)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()


    # Find epilines corresponding to points in first image, and draw the lines on second image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_true)
    lines2 = lines2.reshape(-1, 3)
    img1, img2 = drawlines(image2, image1, lines2, pts2, pts1)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()






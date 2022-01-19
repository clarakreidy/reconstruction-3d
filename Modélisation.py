# -*- coding: utf-8 -*-
"""
Created on 2 Dec 2020

@author: chatoux
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy import ndimage
import glob


def CameraCalibration():
    # Définir les critères d'arrêt. Nous nous arrêtons soit lorsqu'une précision est atteinte, soit lorsque
    # nous avons terminé un certain nombre d'itérations.
    # nous avons choisi 30 itérations et une précision de 0.001
    # nous indiquons à l'algorithme que nous nous soucions à la fois du nombre d'itérations et de la précision (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER).
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    ############ to adapt ##########################
    objp = np.zeros((8 * 5, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)
    # multiplier les valeurs des 2 premières colones par 30
    # taille en mm des carreaux
    objp[:, :2] *= 30
    #################################################
    # Tableaux pour stocker les points d'objet et les points d'image de toutes les images.
    objpoints = []  # Stocker les vecteurs des points 3D pour toutes les images de l'échiquier (cadre de coordonnées mondiales)
    imgpoints = []  # Stocker les vecteurs de points 2D pour toutes les images de l'échiquier (cadre de coordonnées de la caméra)
    ############ to adapt ##########################
    images = glob.glob('Images/chess/ImagesWeTook/*.jpg')
    #################################################
    for fname in images:
        # img = cv.imread(fname)
        img = cv.pyrDown(cv.imread(fname))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Trouvez les coins sur l'échiquier
        ############ to adapt ##########################
        ret, corners = cv.findChessboardCorners(gray, (8, 5), None)
        #################################################
        print(ret)
        # Si les coins sont trouvés par l'algorithme, ajoutez les points de l'objet et les points de l'image.
        if ret == True:
            objpoints.append(objp)
            # affiner l'emplacement du coin (avec une précision inférieure au pixel) en fonction des critères.
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Dessinez les coins de l'échiquier
            ############ to adapt ##########################
            cv.drawChessboardCorners(img, (8, 5), corners2, ret)
            #################################################
            cv.namedWindow('img', 0)
            cv.imshow('img', img)
            cv.waitKey(500)
    cv.destroyAllWindows()

    # Effectuer l'étalonnage de la caméra pour obtenir la matrice de la caméra, les coefficients de distorsion, les vecteurs de rotation et de translation.
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('camraMatrix\n', mtx)
    print('dist\n', dist)

    ############ to adapt ##########################
    img = cv.pyrDown(cv.imread('Images/chess/ImagesWeTook/20220119_102941.jpg'))
    #################################################
    h, w = img.shape[:2]
    # obtenir une matrice de caméra optimale et une région d'intérêt rectangulaire
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print('newcameramtx\n', newcameramtx)

    # trouver une fonction de mappage de l'image déformée à l'image non déformée
    # puis les re-mapper pour supprimer la déformation
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # recadrer l'image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.namedWindow('img', 0)
    cv.imshow('img', dst)
    cv.waitKey(0)
    ############ to adapt ##########################
    cv.imwrite('Images/calibresultM.png', dst)
    #################################################

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    return newcameramtx


def DepthMapfromStereoImages():
    ############ to adapt ##########################
    imgL = cv.pyrDown(cv.imread('Images/peluches/20220119_120515.jpg'))
    imgR = cv.pyrDown(cv.imread('Images/peluches/20220119_120507.jpg'))
    #################################################
    # 
    window_size = 3
    min_disp = 105
    num_disp = 112 - min_disp

    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=window_size,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=1,
                                  uniquenessRatio=15,
                                  speckleWindowSize=0,
                                  speckleRange=2,
                                  preFilterCap=63,
                                  mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)

    #
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    # présenter la figure en 3D de façon que la profondeur dépend de la couleur
    plt.figure('3D')
    plt.imshow((disparity - min_disp) / num_disp, 'gray')
    plt.colorbar()
    plt.show()


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 2)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def StereoCalibrate(Cameramtx):
    ############ to adapt ##########################
    img1 = cv.pyrDown(cv.imread('Images/peluches/20220119_120515.jpg', 0))
    img2 = cv.pyrDown(cv.imread('Images/peluches/20220119_120507.jpg', 0))
    #################################################
    # opencv 4.5
    sift = cv.SIFT_create()
    # opencv 3.4
    # sift = cv.xfeatures2d.SURF_create()
    # trouver les points clés et les descripteurs avec SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    ############## Using FLANN matcher ##############
    # Chaque point clé de la première image est associé à un certain nombre de
    # points clés de la deuxième image. k=2 signifie que l'on garde les 2 meilleures correspondances
    # pour chaque point clé (les meilleures correspondances = celles qui ont la plus petite
    # la plus petite mesure de distance).
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # dessiner les points clefs qui match entre les 2 images avec
    # flag = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS = 2
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    # Show detected matches
    plt.imshow(img3)
    plt.show()

    # Calcule la matrice essentielle à partir des points correspondants dans deux images.
    E, maskE = cv.findEssentialMat(pts1, pts2, Cameramtx, method=cv.FM_LMEDS)
    print('E\n', E)

    # Récupère la rotation et la translation de la caméra  à partir de la matrice essentielle
    # et des points correspondants dans deux images.
    retval, R, t, maskP = cv.recoverPose(E, pts1, pts2, Cameramtx, maskE)
    print('R\n', R)
    print('t\n', t)

    # Calcule une matrice fondamentale à partir des points correspondants dans deux images.
    F, maskF = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    print('F\n', F)

    # Calcul de la matrice fondamentale à partir de la matrice essentielle
    ex = np.array([
        [0, -0, -0],
        [0, 0, -1],
        [0, 1, 0],
    ])
    FT = np.matmul(ex, Cameramtx)
    print('FT\n', FT)

    return pts1, pts2, F, maskF, FT, maskE


def EpipolarGeometry(pts1, pts2, F, maskF, FT, maskE):
    ############ to adapt ##########################
    img1 = cv.pyrDown(cv.imread('Images/peluches/20220119_120515.jpg', 0))
    img2 = cv.pyrDown(cv.imread('Images/peluches/20220119_120507.jpg', 0))
    #################################################
    r, c = img1.shape

    # trouver les points clés et les descripteurs avec SIFT
    # en provenance de la fonction StereoCalibrate
    pts1F = pts1[maskF.ravel() == 1]
    pts2F = pts2[maskF.ravel() == 1]

    # Trouver les lignes épipolaires correspondant aux points de l'image de droite (deuxième image)
    # avec la matrice fondamentale F
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    # dessinant ses lignes sur l'image de droite
    img5, img6 = drawlines(img1, img2, lines1, pts1F, pts2F)
    # Trouver les lignes épipolaires correspondant aux points de l'image de gauche (première image)
    # avec la matrice fondamentale F
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    # dessinant ses lignes sur l'image de droite
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    plt.figure('Fright')
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img6)
    plt.savefig('Fright.png')
    plt.figure('Fleft')
    plt.subplot(121), plt.imshow(img4)
    plt.subplot(122), plt.imshow(img3)
    plt.savefig('Fleft.png')

    # On ne sélecte que les points dont le masque est égale à 1
    pts1 = pts1[maskE.ravel() == 1]
    pts2 = pts2[maskE.ravel() == 1]

    # Trouver les lignes épipolaires correspondant aux points de l'image de droite (deuxième image)
    # avec la matrice fondamentale TF (théorique)
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, FT)
    lines1 = lines1.reshape(-1, 3)
    img5T, img6T = drawlines(img1, img2, lines1, pts1, pts2)
    plt.figure('FTright')
    plt.subplot(121), plt.imshow(img5T)
    plt.subplot(122), plt.imshow(img6T)

    # Trouver les lignes épipolaires correspondant aux points de l'image de droite (deuxième image)
    # avec la matrice fondamentale TF (théorique)
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, FT)
    lines2 = lines2.reshape(-1, 3)
    img3T, img4T = drawlines(img2, img1, lines2, pts2, pts1)
    plt.figure('FTleft')
    plt.subplot(121), plt.imshow(img4T)
    plt.subplot(122), plt.imshow(img3T)
    plt.show()

    # Calcule une transformation de rectification pour une caméra stéréo non calibrée.
    # rectifier les images, produire les homographies : H1 (gauche) et H2 (droite)
    retval, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, (c, r))
    print(H1)
    print(H2)

    # calculer la matrice de transformation de la perspective et la déformation de img1 en H1 et de img2 en H2
    im_dst1 = cv.warpPerspective(img1, H1, (c, r))
    im_dst2 = cv.warpPerspective(img2, H2, (c, r))
    cv.namedWindow('left', 0)
    cv.imshow('left', im_dst1)
    cv.namedWindow('right', 0)
    cv.imshow('right', im_dst2)
    cv.waitKey(0)


if __name__ == "__main__":
    # cameraMatrix = CameraCalibration()

    cameraMatrix = np.array([
        [1.50740955e+03, 0.00000000e+00, 9.99524372e+02],
        [0.00000000e+00, 1.50792468e+03, 7.65799177e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # dist = [[0, 0, 0, 0, 0]]

    #pts1, pts2, F, maskF, FT, maskE = StereoCalibrate(cameraMatrix)

    #EpipolarGeometry(pts1, pts2, F, maskF, FT, maskE)

    DepthMapfromStereoImages()

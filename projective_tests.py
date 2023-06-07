# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:43:18 2023

@author: pgbaz
"""


import json
import math


deg2rad = math.pi / 180.0
rad2deg = 180.0 / math.pi

class GeoCoord():
  def __init__(self, lat, long, alt):
    self.latitude  = lat   # deg
    self.longitude = long  # deg
    self.altitude  = alt   # m

class Observation():
  def __init__(self, az, el, ObserverGC):
    self.az  = az        # deg
    self.el = el         # deg
    self.ObserverGC = ObserverGC # GeoCoord


def main():

  # An airplane at Aeroporto Urbe
  latitudePlane  = 41.95232550018577
  longitudePlane = 12.505142833005202
  altitudePlane = 3000.0
  planeGC = GeoCoord(latitudePlane, longitudePlane, altitudePlane)
  posPlane = getPointInEarthFrameFromGeoCoord(planeGC)

  # First observer location: Settecamini
  latObserver1  = 41.9401565969652
  longObserver1 = 12.621029627805704
  altObserver1 = 0.0  # altitude Observer1 = ?
  Observer1GC = GeoCoord(latObserver1, longObserver1, altObserver1)
  pos7c = getPointInEarthFrameFromGeoCoord(Observer1GC)

  # Second observer location
  latColosseo = 41.89014792072482
  longColosseo = 12.492339876376782

  # To West
  latFiumicino = 41.80582938616291
  longFiumicino = 12.249879780095494
  # To East
  latSubiaco = 41.94367459090483
  longSubiaco = 13.056640155774263
  # To North
  latPoggiomirteto = 42.269958390187576
  longPoggioMirteto = 12.682771249873289
  # To South
  latCampoleone = 41.64334897069568
  longCampoleone = 12.65955145019369

  latObserver2  = latPoggiomirteto
  longObserver2 = longPoggioMirteto
  altObserver2 = 0.0  # altitude altObserver2 = ?
  Observer2GC = GeoCoord(latObserver2, longObserver2, altObserver2)
  posObserver2 = getPointInEarthFrameFromGeoCoord(Observer2GC)

  uap = []
  planes = getPlaneList()
  samplePlane = planeGC # planes[1]
  observation1Az, observation1El = getAzElFromObserver(Observer1GC, samplePlane)
  observation1 = Observation(observation1Az, observation1El, Observer1GC)
  compatible, idx = searchPlane(observation1, planes)

  if (compatible):
    print("Index ", idx, ": ", planes[idx].latitude, planes[idx].longitude, planes[idx].altitude)
  else:
    matchFound, compatible, doUpdate, idx = searchUAP(observation1, uap)
    if (matchFound == True):
      if ((doUpdate == True) and (compatible == False)):
        print("Same observer reporting same observation: just update the state")
        uap[idx] = observation1

      if ((compatible == True) and (doUpdate == False)):
        # UAP confirmed !
        print("UAP confirmed! Index = ", idx, ": ", uap[idx].az, uap[idx].el, uap[idx].ObserverGC.latitude, uap[idx].ObserverGC.longitude, uap[idx].ObserverGC.altitude)
    else:
      print("No matching airplane or Unidentified Aerial Phenomena (UAP). Adding candidate UAP to the UAP list")
      uap.append(observation1)
      printUAPList(uap)


  compatible, idx = searchPlane(observation1, planes)

  if (compatible):
    print("Index ", idx, ": ", planes[idx].latitude, planes[idx].longitude, planes[idx].altitude)
  else:
    matchFound, compatible, doUpdate, idx = searchUAP(observation1, uap)
    if (matchFound == True):
      if ((doUpdate == True) and (compatible == False)):
        print("Same observer reporting same observation: just update the state")
        uap[idx] = observation1

      if ((compatible == True) and (doUpdate == False)):
        # UAP confirmed !
        print("UAP confirmed! Index = ", idx, ": ", uap[idx].az, uap[idx].el, uap[idx].ObserverGC.latitude, uap[idx].ObserverGC.longitude, uap[idx].ObserverGC.altitude)
    else:
      print("No matching airplane or Unidentified Aerial Phenomena (UAP). Adding candidate UAP to the UAP list")
      uap.append(observation1)
      printUAPList(uap)


  observation2Az, observation2El = getAzElFromObserver(Observer2GC, samplePlane)
  observation2 = Observation(observation2Az, observation2El, Observer2GC)
  compatible, idx = searchPlane(observation2, planes)

  if (compatible):
    print("Index ", idx, ": ", planes[idx].latitude, planes[idx].longitude, planes[idx].altitude)
  else:
    matchFound, compatible, doUpdate, idx = searchUAP(observation2, uap)
    if (matchFound == True):
      if ((doUpdate == True) and (compatible == False)):
        print("Same observer reporting same observation: just update the state")
        uap[idx] = observation1

      if ((compatible == True) and (doUpdate == False)):
        # UAP confirmed !
        print("UAP confirmed! Index = ", idx, ": ", uap[idx].az, uap[idx].el, uap[idx].ObserverGC.latitude, uap[idx].ObserverGC.longitude, uap[idx].ObserverGC.altitude)
    else:
      print("No matching airplane or Unidentified Aerial Phenomena (UAP). Adding candidate UAP to the UAP list")
      uap.append(observation1)
      printUAPList(uap)


  return

  # ONLY FOR TEST: generate observation (az, el) from Observer 1 position
  observer1PlaneAz, observer1PlaneEl = generateTestObservation(Observer1GC, planeGC, "Observer1")

  # ONLY FOR TEST: generate observation (az, el) from Observer 2 position
  observer2PlaneAz, observer2PlaneEl = generateTestObservation(Observer2GC, planeGC, "Campoleone")

  compatible = checkAirplane(observer1PlaneAz, observer1PlaneEl, Observer1GC, planeGC)
  print("Obs1 compatible with Flighradar = ", compatible)

  compatible = checkAirplane(observer2PlaneAz, observer2PlaneEl, Observer2GC, planeGC)
  print("Obs2 compatible with Flighradar = ", compatible)

  compatible, P0 = checkUAP(observer1PlaneAz, observer1PlaneEl, Observer1GC, observer2PlaneAz, observer2PlaneEl, Observer2GC, planeGC)
  print("compatible with Ufo = ", compatible)

  xairplane = altitudePlane * math.cos(observer1PlaneEl) * math.cos(observer1PlaneAz)
  yairplane = altitudePlane * math.cos(observer1PlaneEl) * math.sin(observer1PlaneAz)
  zairplane = altitudePlane * math.sin(observer1PlaneEl)
  xpVanish, ypVanish, x1q, y1q, z1q, valid = transformPoint(xairplane, yairplane, zairplane, phoneRotationQuaternion, phoneRotationQuaternionConj, sx, sy, tx, ty)
  circle = plt.Circle((xpVanish, ypVanish), 20.0, color='b')
  plt.gca().add_patch(circle)
  plt.show()

def makeArray(r, c):
  rows, cols = (r, c)
  if ((r > 1) and (c == 1)):
    C = [[0.0] for j in range(rows)]
  if ((r == 1) and (c > 1)):
    C = [0.0 for i in range(cols)]
  if((r > 1) and (c > 1)):
    C = [[0.0 for i in range(cols)] for j in range(rows)]
  if((r == 1) and (c == 1)):
    C = [0.0]
  return C


def checkInputSize(A):
  if hasattr(A[0], '__len__'):
    colsA = len(A[0])
    rowsA = len(A)
  else:
    rowsA = 1
    if hasattr(A, '__len__'):
      colsA = len(A)
    else:
      colsA = 1
  return rowsA, colsA


def transposeMat(A):
  rowsA, colsA = checkInputSize(A)
  if ((colsA > 1) and (rowsA > 1)):
    C = makeArray(colsA, rowsA)
    for row in range(rowsA): 
      for col in range(colsA):
        C[col][row] = A[row][col]
  elif ((colsA == 1) and (rowsA > 1)):
    C = makeArray(colsA, rowsA)
    for row in range(rowsA): 
        C[row] = A[row][0]
  elif ((colsA > 1) and (rowsA == 1)):
    C = makeArray(colsA, rowsA)
    for col in range(colsA): 
        C[col][0] = A[col]
  else:
    print("Che te sei bevuto ? (transposeMat)")
    C = 0
  return C


def mulMat(A, B):
  rowsA, colsA = checkInputSize(A)
  rowsB, colsB = checkInputSize(B)
  if (colsA == rowsB):
    C = makeArray(rowsA, colsB)
    for row in range(rowsA): 
      for col in range(colsB):
        for elt in range(rowsA):
          C[row][col] += A[row][elt] * B[elt][col]
  else:
    print("Che te sei bevuto ? (mulMat)")
    C = 0
  return C


def matByScalar(A, s):
  rowsA, colsA = checkInputSize(A)
  C = makeArray(rowsA, colsA)
  if((rowsA > 1) and (colsA > 1)):
    for row in range(rowsA): 
      for col in range(colsA):
        C[row][col] = A[row][col] * s
  elif ((rowsA > 1) and (colsA == 1)):
    for row in range(rowsA): 
      C[row][0] = A[row][0] * s
  elif ((colsA > 1) and (rowsA == 1)):
    for col in range(colsA):
      C[col] = A[col] * s
  else:
    print("Che te sei bevuto ? (matByScalar)")
  return C


def vecDot(A, B):
  rowsA, colsA = checkInputSize(A)
  rowsB, colsB = checkInputSize(B)
  resDot = 0.0
  if ((rowsA == rowsB) and (colsA == 1) and (colsB == 1)):
    if (rowsA >= 1):
      for row in range(rowsA):
        resDot += A[row][0] * B[row][0]
  elif ((colsA == colsB) and (rowsA == 1) and (rowsB == 1)):
    if (colsA >= 1):
      for col in range(colsA):
        resDot += A[col] * B[col]
  else:
    print("Che te sei bevuto ? (dot)")
  return resDot


def matSum(A, B):
  rowsA, colsA = checkInputSize(A)
  rowsB, colsB = checkInputSize(B)
  if ((rowsA == rowsB) and (colsA == colsB)):
    C = makeArray(rowsA, colsA)
    if((rowsA > 1) and (colsA > 1)):
      for row in range(rowsA): 
        for col in range(colsB):
          C[row][col] = A[row][col] + B[row][col]
    elif((rowsA > 1) and (colsA == 1)):
      for row in range(rowsA): 
        C[row][0] = A[row][0] + B[row][0]
    elif((colsA > 1) and (rowsA == 1)):
      for col in range(colsA):
        C[col] = A[col] + B[col]
  else:
    print("Che te sei bevuto ? (vecSum)")
  return C


def vecCross(A, B):
  rowsA, colsA = checkInputSize(A)
  rowsB, colsB = checkInputSize(B)
  if ((colsA == 3) and (colsB == 3)):
    C = makeArray(colsA, colsB)
    C[0] = A[1] * B[2] - A[2] * B[1]
    C[1] = A[2] * B[0] - A[0] * B[2]
    C[2] = A[0] * B[1] - A[1] * B[0]
  else:
    print("Che te sei bevuto ? (cross)")
    C = 0
  return C


def getPlaneList():
  f = open('api_example_rome_box.json',) # Opening list of airplanes
  data = json.load(f) # returns JSON object as a dictionary
  planes = []

  # Iterating through the airplane list, discarding airplanes with invalid information
  for i in data['states']:
    if (isinstance(i[5], float) and isinstance(i[6], float) and isinstance(i[7], float)):
      planes.append(GeoCoord(i[6], i[5], i[7]))

  f.close()
  return planes


def searchPlane(observation1, planes):
  compatible = False
  i = 0
  for i in range(len(planes)):
    comp = checkAirplane(observation1, planes[i])
    compatible = compatible or comp
    if (comp == True):
      break
  return compatible, i


def searchUAP(observation1, uap):
  compatible = False
  doUpd = False
  matchFound = False
  i = 0
  for i in range(len(uap)):
    print(uap[i].az, uap[i].el, uap[i].ObserverGC.latitude, uap[i].ObserverGC.longitude, uap[i].ObserverGC.altitude)
    compatible, doUpd = checkUAP(observation1, uap[i])
    if ((doUpd == True) or (compatible == True)):
      matchFound = True
      break
  return matchFound, compatible, doUpd, i


def printUAPList(uap):
  for i in range(len(uap)):
    print(uap[i].az, uap[i].el, uap[i].ObserverGC.latitude, uap[i].ObserverGC.longitude, uap[i].ObserverGC.altitude)
  return


def generateTestObservation(observerGC, planeGC, posName):
  posObserver = getPointInEarthFrameFromGeoCoord(observerGC)
  posPlane = getPointInEarthFrameFromGeoCoord(planeGC)
  TransformMatrix = getTransformationMatrix(observerGC)
  relPos = matSum(posPlane, matByScalar(posObserver, -1.0))
  posPlaneInObserverFrame = transposeMat(mulMat(TransformMatrix, transposeMat(relPos)))
  pInObsX = posPlaneInObserverFrame[0]
  pInObsY = posPlaneInObserverFrame[1]
  pInObsZ = posPlaneInObserverFrame[2]
  observerPlaneAz = math.atan2(pInObsY, pInObsX)
  observerPlaneEl = math.atan2(pInObsZ, math.sqrt(pInObsX * pInObsX + pInObsY * pInObsY))

  origin = makeArray(1, 3)
  origin[0] = 0.0
  origin[1] = 0.0
  origin[2] = 0.0
  dist1 = pointPointDistance(posPlaneInObserverFrame, origin)
  print("Position of the airplane in local ", posName, " frame = ", posPlaneInObserverFrame)
  print("Dist = ", dist1, "Az = ", observerPlaneAz * rad2deg, "El = ", observerPlaneEl * rad2deg)

  return observerPlaneAz,observerPlaneEl


def getAzElFromObserver(observerGC, planeGC):
  posObserver = getPointInEarthFrameFromGeoCoord(observerGC)
  posPlane = getPointInEarthFrameFromGeoCoord(planeGC)
  TransformMatrix = getTransformationMatrix(observerGC)
  relPos = matSum(posPlane, matByScalar(posObserver, -1.0))
  posPlaneInObserverFrame = transposeMat(mulMat(TransformMatrix, transposeMat(relPos)))
  pInObsX = posPlaneInObserverFrame[0]
  pInObsY = posPlaneInObserverFrame[1]
  pInObsZ = posPlaneInObserverFrame[2]
  planeAz = math.atan2(pInObsY, pInObsX)
  planeEl = math.atan2(pInObsZ, math.sqrt(pInObsX * pInObsX + pInObsY * pInObsY))

  return planeAz,planeEl


def getPointInEarthFrameFromGeoCoord(GeoCoord):
  """
  Return the coordinates of the point in the earth frame given latitude and longitude

  Parameters
  ----------
  lat : latitude angle.  Unit: deg
  log : longitude angle. Unit: deg

  Returns
  -------
  x : coordinate x of the point in the earth frame. Unit: m
  y : coordinate y of the point in the earth frame. Unit: m
  z : coordinate z of the point in the earth frame. Unit: m

  """
  r = 6373044.737 # earth radius. Unit: meters
  lat = GeoCoord.latitude  * deg2rad
  lon = GeoCoord.longitude * deg2rad
  r = r + GeoCoord.altitude
  x = r * math.cos(lat) * math.cos(lon)
  y = r * math.cos(lat) * math.sin(lon)
  z = r * math.sin(lat)
  P = makeArray(1, 3)
  P[0] = x
  P[1] = y
  P[2] = z
  return P


def pointPointDistance(P1, P2):
  q1 = (P1[0]-P2[0]) * (P1[0]-P2[0])
  q2 = (P1[1]-P2[1]) * (P1[1]-P2[1])
  q3 = (P1[2]-P2[2]) * (P1[2]-P2[2])
  d = math.sqrt(q1 + q2 + q3)
  return d


def lineLineDistance(az1, el1, az2, el2, Observer2InFrame1):
  """
  Compute the distance between 2 lines in 3D space.
  Line 1 passes in the origin of the observer 1 assuming that 
  the main frame is placed on the Observer 1 (0, 0, 0)
  Line 2 passes in the origin of the observer 2.
  Observer 2 is in position "Observer2InFrame1" in the frame of the Observer1.
  Compute method: in order to be the min distance vector, the vector connecting 
  a generic point on Line 1 and a generic point on Line 2, shall be orthogonal 
  to both the vector identifying the Line 1 and the vector identifying the Line 2.
  The distance computed can be used as the distance between 2 observations.

  Parameters
  ----------
  az1 : float
    Azimuth of the observation made by observer 1.
  el1 : float
    Elevation of the observation made by observer 1.
  az2 : float
    Azimuth of the observation made by observer 2.
  el2 : float
    Elevation of the observation made by observer 2.
  Observer2InFrame1 : float array
    Contains x, y, z coordinates of the Observer 2 in the frame of Observer 1.

  Returns
  -------
  dist : float
    distance between the 2 lines
  P0 : float array
    Contains x, y, z coordinates of the estimate of Observation 1.
  P1 : float array
    Contains x, y, z coordinates of the estimate of Observation 2.

  """
  x2 = Observer2InFrame1[0]
  y2 = Observer2InFrame1[1]
  z2 = Observer2InFrame1[2]
  
  b =      math.cos(el1) * math.cos(el2) * math.cos(az1) * math.cos(az2)
  b = b  + math.cos(el1) * math.cos(el2) * math.sin(az1) * math.sin(az2)
  b = b  + math.sin(el1) * math.sin(el2)
  c = x2 * math.cos(el1) * math.cos(az1) + y2 * math.cos(el1) * math.sin(az1) + z2 * math.sin(el1)
  d = x2 * math.cos(el2) * math.cos(az2) + y2 * math.cos(el2) * math.sin(az2) + z2 * math.sin(el2)
  
  # In general some check should be necessary to avoid div by zero.
  # In reality the fact that the observers are distinct should always
  # prevent the div by zero conditon. Check skipped
  k2 = (d - b * c) / (b * b - 1)
  k1 = k2 * b + c

  xr0 = k1 * math.cos(el1) * math.cos(az1)
  yr0 = k1 * math.cos(el1) * math.sin(az1)
  zr0 = k1 * math.sin(el1)
  #P0 = np.array([xr0, yr0, zr0])
  P0 = makeArray(1, 3)
  P0[0] = xr0
  P0[1] = yr0
  P0[2] = zr0

  xr1 = k2 * math.cos(el2) * math.cos(az2) + x2
  yr1 = k2 * math.cos(el2) * math.sin(az2) + y2
  zr1 = k2 * math.sin(el2) + z2
  #P1 = np.array([xr1, yr1, zr1])
  P1 = makeArray(1, 3)
  P1[0] = xr1
  P1[1] = yr1
  P1[2] = zr1
  
  dist = pointPointDistance(P0, P1)
  return dist, P0, P1


def pointLineDistance(az, el, P0):
  """
  Compute the distance between a point and a line in 3D space
  The Line passes in the position of the Observer (origin of the frame),
  assuming that the main frame placed in the Observer position i.e. (0, 0, 0)
  The point P0 is the 3D point from Flightradar which identifies the airplane 
  position in the frame of the Observer.
  Compute method: in order to be the min distance vector, the vector connecting 
  a generic point on the Line and the input point, shall be orthogonal to the 
  vector identifying the Line.

  Parameters
  ----------
  az : float
    Azimuth of the observation made by Observer. Unit: rad
  el : float
    Elevation of the observation made by Observer.  Unit: rad
  P0 : float array
    Contains x, y, z coordinates of the point.  Unit: m

  Returns
  -------
  dist : float
    distance between the point P0 and the line (airplane - observation disatance)
  """
  x0 = P0[0]
  y0 = P0[1]
  z0 = P0[2]

  k = x0 * math.cos(el) * math.cos(az) + y0 * math.cos(el) * math.sin(az) + z0 * math.sin(el)
  
  x1 = k * math.cos(el) * math.cos(az)
  y1 = k * math.cos(el) * math.sin(az)
  z1 = k * math.sin(el)
  #P1 = np.array([x1, y1, z1])
  P1 = makeArray(1, 3)
  P1[0] = x1
  P1[1] = y1
  P1[2] = z1

  dist = pointPointDistance(P0, P1)
  
  return dist


def checkUAP(observation1, observation2):
  """
  Check whether 2 observations are compatible with each other.
  Supposed to run on cloud
 
  Parameters
  ----------
  observation1 : Class Observation.
    Contains:
      azimuth angle of the airplane in the frame of the observer (unit deg).
      elevation angle of the airplane in the frame of the observer (unit deg).
      observer Geo Coord: latitude (unit deg), longitude (unit deg) and altitude (unit m)
  observation2 : Class Observation.
    Contains:
      azimuth angle of the airplane in the frame of the observer (unit deg).
      elevation angle of the airplane in the frame of the observer (unit deg).
      observer Geo Coord: latitude (unit deg), longitude (unit deg) and altitude (unit m)

  Returns
  -------
  compatible : bool
    True if observation 1 and observation 2 are compatible.
  """
  compatible = False
  doUpdate = False

  TransformMatrix1 = getTransformationMatrix(observation1.ObserverGC)
  TransformMatrix2 = getTransformationMatrix(observation2.ObserverGC)
  TransformMatrix3 = mulMat(TransformMatrix1, transposeMat(TransformMatrix2))

  posObserver1 = getPointInEarthFrameFromGeoCoord(observation1.ObserverGC)
  posObserver2 = getPointInEarthFrameFromGeoCoord(observation2.ObserverGC)

  minObserversDistance = 20.0

  # Avoid confirming UAP from the same observer
  # Better approach would be checking the observer position AND the observation parameters
  # by now only the observer position is checked (if the same observer reports more than 1 object the second is discarded!)
  if (pointPointDistance(posObserver1, posObserver2) < minObserversDistance):
    doUpdate = True
  else:
    obs2InObs1 = matSum(posObserver2, matByScalar(posObserver1, -1.0))
    #o2InO1 = np.matmul(TransformMatrix1, obs2InObs1)
    o2InO1 = transposeMat(mulMat(TransformMatrix1, transposeMat(obs2InObs1)))
    # Build the vector associated to the direction of the plane in the frame of Observer 2
    d = makeArray(1, 3)
    d[0] = math.cos(observation2.el)*math.cos(observation2.az)
    d[1] = math.cos(observation2.el)*math.sin(observation2.az)
    d[2] = math.sin(observation2.el)
    moduleD = math.sqrt(vecDot(d, d))
    d = matByScalar(d, (1.0 / moduleD))

    # Transform the vector associated to the direction of the plane to the frame of Observer 1
    dInO1 = transposeMat(mulMat(TransformMatrix3, transposeMat(d)))
    # Compute the updated azimuth and elevation in the frame of Observer 1
    ro = math.sqrt(dInO1[0]*dInO1[0] + dInO1[1]*dInO1[1] + dInO1[2]*dInO1[2])
    el = math.asin(dInO1[2] / ro)
    az = math.atan2(dInO1[1], dInO1[0])
    # Compute the min distance of the two lines in the frame of Observer 1
    lineLineDist_2, P0_2, P1_2 = lineLineDistance(observation1.az, observation1.el, az, el, o2InO1)

    # Let assume by now a linear increase of the error with the distance of the object from the observers 
    # To get an error estimate we assume an increase factor equal to 1e-3 (1m every 1000m)
    observer1Origin = makeArray(1, 3)
    observer1Origin[0] = 0.0
    observer1Origin[1] = 0.0
    observer1Origin[2] = 0.0
    ufoDist1 = pointPointDistance(P0_2, observer1Origin)
    ufoDist2 = pointPointDistance(P1_2, observer1Origin)
    ufoDist3 = pointPointDistance(P0_2, o2InO1)
    ufoDist4 = pointPointDistance(P1_2, o2InO1)

    ufoDist = ufoDist1
    if (ufoDist2 > ufoDist):
      ufoDist = ufoDist2
    if (ufoDist3 > ufoDist):
      ufoDist = ufoDist3
    if (ufoDist4 > ufoDist):
      ufoDist = ufoDist4

    coeff = 1e-3
    maxErr = coeff * ufoDist
    if(lineLineDist_2 < maxErr):
      compatible = True
    else:
      compatible = False
  
  return compatible, doUpdate


def checkAirplane(observation1, geoCoordAirplane):
  """
  Check whether 1 observations is compatible with Flightradar.
  Supposed to run on the mobile (probably)
  Steps:  
    1) get local 3D coord of Airplane in the frame of the Observer
    2) compute the distance between observation and airplane (pointLineDistance)
    3) check if the observation is compatible with the Airplane
 
  Parameters
  ----------
  observation1 : Class Observation.
    Contains azimuth angle of the airplane in the frame of the observer (unit deg).
    elevation angle of the airplane in the frame of the observer (unit deg).
    observer Geo Coord: latitude (unit deg), longitude (unit deg) and altitude (unit m)
  GeoCoordAirplane : class GeoCoord()
    Contains latitude (unit deg), longitude (unit deg) and altitude (unit m) from Flightradar.

  Returns
  -------
  compatible : bool
    True if observation 1 and the position from Flightradar are compatible.

  """
  posObserver = getPointInEarthFrameFromGeoCoord(observation1.ObserverGC)
  posPlane = getPointInEarthFrameFromGeoCoord(geoCoordAirplane)
  TransformMatrix = getTransformationMatrix(observation1.ObserverGC)
  #posPlaneLoc = np.matmul(TransformMatrix, (posPlane - posObserver))
  relPos = matSum(posPlane, matByScalar(posObserver, -1.0))
  posPlaneLoc = transposeMat(mulMat(TransformMatrix, transposeMat(relPos)))

  ptLineDist = pointLineDistance(observation1.az, observation1.el, posPlaneLoc)

  # Let assume by now a linear increase of the error with the distance with increase factor equal to 1e-3
  #observerOrigin = np.array([0.0, 0.0, 0.0])
  observerOrigin = makeArray(1, 3)
  observerOrigin[0] = 0.0
  observerOrigin[1] = 0.0
  observerOrigin[2] = 0.0
  airplaneDist = pointPointDistance(posPlaneLoc, observerOrigin)
  coeff = 1e-3
  maxErr = coeff * airplaneDist
  if(ptLineDist < maxErr):
    compatible = True
  else:
    compatible = False

  return compatible


def getTransformationMatrix(GeoCoordOfOrigin):
  """
  The computation of the local frame associated to the input point is done using
  the definition of the gradient and normal vectors to a curve described as f(x,y,z) = 0
  For the part of code related to the construction of the transformation matrix 
  see https://vcg.isti.cnr.it/~cignoni/SciViz1819/SciViz_05_Trasformazioni.pdf
  """
  P0 = getPointInEarthFrameFromGeoCoord(GeoCoordOfOrigin)
  
  # Compute a vector orthogonal to the surface of the sphere, pointing outside
  gradf = makeArray(1, 3)
  gradf[0] = 2.0 * P0[0]
  gradf[1] = 2.0 * P0[1]
  gradf[2] = 2.0 * P0[2]
  moduleV = math.sqrt(vecDot(gradf, gradf))
  v = matByScalar(gradf, (1.0 / moduleV))

  # East is ortogonal to the normal to the surface without Z component
  # that is a vector tangent to the circle at z = z0 (section of the sphere at z=z0)
  east = makeArray(1, 3)
  east[0] = -2.0 * P0[1]
  east[1] =  2.0 * P0[0]
  east[2] =  0.0
  moduleE = math.sqrt(vecDot(east, east))
  e = matByScalar(east, (1.0 / moduleE))

  # North (n vector) is orthogonal to the Normal to the sphere (v - vertical vector) and to the East (e vector)
  north = vecCross(v, e)
  moduleN = math.sqrt(vecDot(north, north))
  n = matByScalar(north, (1.0 / moduleN))

  TransformMatrix = makeArray(3, 3)

  TransformMatrix[0][0] = e[0]
  TransformMatrix[0][1] = e[1]
  TransformMatrix[0][2] = e[2]
  TransformMatrix[1][0] = n[0]
  TransformMatrix[1][1] = n[1]
  TransformMatrix[1][2] = n[2]
  TransformMatrix[2][0] = v[0]
  TransformMatrix[2][1] = v[1]
  TransformMatrix[2][2] = v[2]
 
  return TransformMatrix


if __name__ == "__main__":
  main()



"""
  A = makeArray(3, 3)
  A[0][0] = 1.0
  A[0][1] = 0.0
  A[0][2] = 0.0
  A[1][0] = 0.0
  A[1][1] = 1.0
  A[1][2] = 0.0
  A[2][0] = 0.0
  A[2][1] = 0.0
  A[2][2] = 1.0
  print ("A = ", A)
  print("transposeMat(A): ", transposeMat(A))

  B = makeArray(3, 1)
  B[0][0] = 1.0
  B[1][0] = 2.0
  B[2][0] = 3.0
  print("B: ", B)
  print("transposeMat(B): ", transposeMat(B))

  C = makeArray(1, 3)
  C[0] = 1.0
  C[1] = 2.0
  C[2] = 3.0
  print("C: ", C)
  print("transposeMat(C): ", transposeMat(C))

  M1 = mulMat(A, B)
  print("A = ", A)
  print("B = ", B)
  print("M1 = ", M1)

  M2 = transposeMat(mulMat(A, transposeMat(C)))
  print("A = ", A)
  print("C = ", C)
  print("M2 = ", M2)

  resDot = vecDot(B, B)
  print("B dot B", resDot)
  D = makeArray(1, 3)
  D[0] = 1.0
  D[1] = 1.0
  D[2] = 1.0
  E = [1.0, 2.0, 3.0]
  print ("D = ", D)
  print ("E = ", E)
  resDot = vecDot(E, D)
  print("E dot D", resDot)

  F = [1.0, 0.0, 0.0]
  G = [0.0, 1.0, 0.0]
  print("F = ", F, "G = ", G)
  resCross = vecCross(E, E)
  print("cross E, E = ", resCross)
  resCross = vecCross(F, G)
  print("cross F, G = ", resCross)
  resCross = vecCross(G, F)
  print("cross G, F = ", resCross)

  H = makeArray(3, 1)
  print("H(3, 1): ", H)
  I = makeArray(3, 3)
  print("I(3, 3): ", I)
  L = makeArray(1, 3)
  print("L(1, 3): ", L)

  res1 = matByScalar(A, 2.0)
  print("A * 2 = ", res1)

  res2 = matByScalar(E, 2.0)
  print("E * 2 = ", res2)

  res3 = matByScalar(B, 2.0)
  print("B * 2 = ", res3 )

  print("matSum(A, A)", matSum(A, A))
  print("matSum(B, B)", matSum(B, B))
  print("matSum(C, C)", matSum(C, C))

  print("matSum(A, -A)", matSum(A, matByScalar(A, -1)))
  print("matSum(B, -B)", matSum(B, matByScalar(B, -1)))
  print("matSum(C, -C)", matSum(C, matByScalar(C, -1)))

  print("matSum(-A, A)", matSum(matByScalar(A, -1), A))
  print("matSum(-B, B)", matSum(matByScalar(B, -1), B))
  print("matSum(-C, C)", matSum(matByScalar(C, -1), C))

  print("matSum(-F, G)", matSum(matByScalar(F, -1), G))

  return
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:43:18 2023

@author: pgbaz
"""


import json
import numpy as np

deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi


# Geo Coordinates contain latitude, longitude and altitude
class GeoCoord():
  def __init__(self, lat, long, alt):
    self.latitude  = lat   # deg
    self.longitude = long  # deg
    self.altitude  = alt   # m


# An observation is defined by the observed Azimuth, observed Elelvation and observer Geo Coordinates
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
    compatible, doUpdate, idx = searchUAP(observation1, uap)
    if ((compatible == True) or (doUpdate == True)):
      if ((compatible == False) and (doUpdate == True)):
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
    compatible, doUpdate, idx = searchUAP(observation1, uap)
    if ((compatible == True) or (doUpdate == True)):
      if ((compatible == False) and (doUpdate == True)):
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
    compatible, doUpdate, idx = searchUAP(observation2, uap)
    if ((compatible == True) or (doUpdate == True)):
      if ((compatible == False) and (doUpdate == True)):
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


def searchPlane(observation, planes):
  """
  Search in the list of planes, the Aerial object described by the input observation 
  Parameters
  ----------
  observation : Class Observation
  planes : list of Class Observation elements

  Returns
  compatible : bool 
    True if the observation is compatible with one of the planes in the list.
  i : int
    index of the matching plane in the list
  -------
  """
  compatible = False
  i = 0
  for i in range(len(planes)):
    comp = checkAirplane(observation, planes[i])
    compatible = compatible or comp
    if (comp == True):
      break
  return compatible, i


def searchUAP(observation, uap):
  """
  Search in the list of the Unidentified Aerial Phenomena (UAP), the object described by the input observation 
  Parameters
  ----------
  observation : Class Observation
  uap : list of Class Observation elements

  Returns
  compatible : bool 
    True if the observation is compatible with one of the planes in the list.
  doUpd : bool
    True if the same observer is confirming the object multiple times.
  i : int
    index of the matching plane in the list
  -------
  """
  compatible = False
  doUpd = False
  i = 0
  for i in range(len(uap)):
    print(uap[i].az, uap[i].el, uap[i].ObserverGC.latitude, uap[i].ObserverGC.longitude, uap[i].ObserverGC.altitude)
    compatible, doUpd = checkUAP(observation, uap[i])
    if ((doUpd == True) or (compatible == True)):
      break
  return compatible, doUpd, i


def printUAPList(uap):
  for i in range(len(uap)):
    print(uap[i].az, uap[i].el, uap[i].ObserverGC.latitude, uap[i].ObserverGC.longitude, uap[i].ObserverGC.altitude)
  return


def generateTestObservation(observerGC, planeGC, posName):
  posObserver = getPointInEarthFrameFromGeoCoord(observerGC)
  posPlane = getPointInEarthFrameFromGeoCoord(planeGC)
  TransformMatrix = getTransformationMatrix(observerGC)
  relPos = posPlane - posObserver
  posPlaneInObserverFrame = np.matmul(TransformMatrix, relPos)
  pInObsX = posPlaneInObserverFrame[0]
  pInObsY = posPlaneInObserverFrame[1]
  pInObsZ = posPlaneInObserverFrame[2]
  observerPlaneAz = np.arctan2(pInObsY, pInObsX)
  observerPlaneEl = np.arctan2(pInObsZ, np.sqrt(pInObsX * pInObsX + pInObsY * pInObsY))

  dist1 = pointPointDistance(posPlaneInObserverFrame, np.array([0.0, 0.0, 0.0]))
  print("Position of the airplane in local ", posName, " frame = ", posPlaneInObserverFrame)
  print("Dist = ", dist1, "Az = ", observerPlaneAz * rad2deg, "El = ", observerPlaneEl * rad2deg)

  return observerPlaneAz,observerPlaneEl


def getAzElFromObserver(observerGC, planeGC):
  """
  Generates the Azimuth and Elelvation  of the aerial object from the observer position and the plane position 

  Parameters
  ----------
  observerGC : GeoCoord of the observer
  planeGC : GeoCoord of the aerial object

  Returns
  -------
  planeAz : float
    Azimuth of the aerial object in the frame of the Observer. Unit: rad
  planeEl : float
    Elevation of the aerial object in the frame of the Observer. Unit: rad
  """
  posObserver = getPointInEarthFrameFromGeoCoord(observerGC)
  posPlane = getPointInEarthFrameFromGeoCoord(planeGC)
  TransformMatrix = getTransformationMatrix(observerGC)
  relPos = posPlane - posObserver
  posPlaneInObserverFrame = np.matmul(TransformMatrix, relPos)
  pInObsX = posPlaneInObserverFrame[0]
  pInObsY = posPlaneInObserverFrame[1]
  pInObsZ = posPlaneInObserverFrame[2]
  planeAz = np.arctan2(pInObsY, pInObsX)
  planeEl = np.arctan2(pInObsZ, np.sqrt(pInObsX * pInObsX + pInObsY * pInObsY))

  return planeAz, planeEl


def getPointInEarthFrameFromGeoCoord(pointGC):
  """
  Return the coordinates of the point in the earth frame given latitude and longitude

  Parameters
  ----------
  pointGC : GeoCoord of the point

  Returns
  -------
  P : array of float
    index 0 --> x : coordinate x of the point in the earth frame. Unit: m
    index 1 --> y : coordinate y of the point in the earth frame. Unit: m
    index 2 --> z : coordinate z of the point in the earth frame. Unit: m
  """
  r = 6373044.737 # earth radius. Unit: meters
  lat = pointGC.latitude  * deg2rad
  lon = pointGC.longitude * deg2rad
  r = r + pointGC.altitude
  x = r * np.cos(lat) * np.cos(lon)
  y = r * np.cos(lat) * np.sin(lon)
  z = r * np.sin(lat)
  P = np.array([x, y, z])
  return P


def pointPointDistance(P1, P2):
  """
  Euclidean disttance of 2 points. Note: in general not corresponding to the distance on the earth surface

  Parameters
  ----------
  P1 : Coordinates of the first  point. Unit m
  P2 : Coordinates of the second point. Unit m

  Returns
  d : float
    Eucledean distance of the two points. Unit m
  -------
  """
  q1 = (P1[0]-P2[0]) * (P1[0]-P2[0])
  q2 = (P1[1]-P2[1]) * (P1[1]-P2[1])
  q3 = (P1[2]-P2[2]) * (P1[2]-P2[2])
  d = np.sqrt(q1 + q2 + q3)
  return d


def pointLineDistance(az, el, p0):
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
  x0 = p0[0]
  y0 = p0[1]
  z0 = p0[2]

  k = x0 * np.cos(el) * np.cos(az) + y0 * np.cos(el) * np.sin(az) + z0 * np.sin(el)
  
  x1 = k * np.cos(el) * np.cos(az)
  y1 = k * np.cos(el) * np.sin(az)
  z1 = k * np.sin(el)
  p1 = np.array([x1, y1, z1])

  dist = pointPointDistance(p0, p1)
  
  return dist


def lineLineDistance(az1, el1, az2, el2, observer2InFrame1):
  """
  Compute the distance between 2 lines in 3D space.
  Line 1 passes in the origin of the observer 1 assuming that 
  the main frame is placed on the Observer 1 (0, 0, 0)
  Line 2 passes in the origin of the observer 2.
  Observer 2 is in position "observer2InFrame1" in the frame of the Observer1.
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
  x2 = observer2InFrame1[0]
  y2 = observer2InFrame1[1]
  z2 = observer2InFrame1[2]
  
  b =      np.cos(el1) * np.cos(el2) * np.cos(az1) * np.cos(az2)
  b = b  + np.cos(el1) * np.cos(el2) * np.sin(az1) * np.sin(az2)
  b = b  + np.sin(el1) * np.sin(el2)
  c = x2 * np.cos(el1) * np.cos(az1) + y2 * np.cos(el1) * np.sin(az1) + z2 * np.sin(el1)
  d = x2 * np.cos(el2) * np.cos(az2) + y2 * np.cos(el2) * np.sin(az2) + z2 * np.sin(el2)
  
  # In general some check should be necessary to avoid div by zero.
  # In reality the fact that the observers are distinct should always
  # prevent the div by zero conditon. Check skipped
  k2 = (d - b * c) / (b * b - 1)
  k1 = k2 * b + c

  xr0 = k1 * np.cos(el1) * np.cos(az1)
  yr0 = k1 * np.cos(el1) * np.sin(az1)
  zr0 = k1 * np.sin(el1)
  P0 = np.array([xr0, yr0, zr0])

  xr1 = k2 * np.cos(el2) * np.cos(az2) + x2
  yr1 = k2 * np.cos(el2) * np.sin(az2) + y2
  zr1 = k2 * np.sin(el2) + z2
  P1 = np.array([xr1, yr1, zr1])
  
  dist = pointPointDistance(P0, P1)
  return dist, P0, P1


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
  doUpdate : bool
    True if the observation is confirmed by and observer different from the current one  
  NOTE: the function shall never return "compatible" and "doUpdate" both true at the same time
  """
  compatible = False
  doUpdate = False

  TransformMatrix1 = getTransformationMatrix(observation1.ObserverGC)
  TransformMatrix2 = getTransformationMatrix(observation2.ObserverGC)
  TransformMatrix3 = np.matmul(TransformMatrix1, TransformMatrix2.transpose())

  posObserver1 = getPointInEarthFrameFromGeoCoord(observation1.ObserverGC)
  posObserver2 = getPointInEarthFrameFromGeoCoord(observation2.ObserverGC)

  minObserversDistance = 20.0

  # Avoid confirming UAP from the same observer
  # Better approach would be checking the observer position AND the observation parameters
  # by now only the observer position is checked (if the same observer reports more than 1 object the second is discarded!)
  if (pointPointDistance(posObserver1, posObserver2) < minObserversDistance):
    compatible = False
    doUpdate = True
  else:
    obs2InObs1 = posObserver2 - posObserver1
    o2InO1 = np.matmul(TransformMatrix1, obs2InObs1)
    # Build the vector associated to the direction of the plane in the frame of Observer 2
    d = np.array([np.cos(observation2.el)*np.cos(observation2.az), np.cos(observation2.el)*np.sin(observation2.az), np.sin(observation2.el)])
    d = d / np.sqrt(np.dot(d, d))
    # Transform the vector associated to the direction of the plane to the frame of Observer 1
    dInO1 = np.matmul(TransformMatrix3, d)
    # Compute the updated azimuth and elevation in the frame of Observer 1
    ro = np.sqrt(dInO1[0]*dInO1[0] + dInO1[1]*dInO1[1] + dInO1[2]*dInO1[2])
    el = np.arcsin(dInO1[2] / ro)
    az = np.arctan2(dInO1[1], dInO1[0])
    # Compute the min distance of the two lines in the frame of Observer 1
    lineLineDist_2, P0_2, P1_2 = lineLineDistance(observation1.az, observation1.el, az, el, o2InO1)

    # Let assume by now a linear increase of the error with the distance of the object from the observers 
    # To get an error estimate we assume an increase factor equal to 1e-3 (1m every 1000m)
    observer1Origin = np.array([0.0, 0.0, 0.0])
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
  posPlaneLoc = np.matmul(TransformMatrix, (posPlane - posObserver))

  ptLineDist = pointLineDistance(observation1.az, observation1.el, posPlaneLoc)

  # Let assume by now a linear increase of the error with the distance with increase factor equal to 1e-3
  observerOrigin = np.array([0.0, 0.0, 0.0])
  airplaneDist = pointPointDistance(posPlaneLoc, observerOrigin)
  coeff = 1e-3
  maxErr = coeff * airplaneDist
  if(ptLineDist < maxErr):
    compatible = True
  else:
    compatible = False

  return compatible


def getTransformationMatrix(geoCoordOfOrigin):
  """
  The computation of the local frame associated to the input point is done using
  the definition of the gradient and normal vectors to a curve described as f(x,y,z) = 0
  For the part of code related to the construction of the transformation matrix 
  see https://vcg.isti.cnr.it/~cignoni/SciViz1819/SciViz_05_Trasformazioni.pdf
  """
  P0 = getPointInEarthFrameFromGeoCoord(geoCoordOfOrigin)
  
  # Compute a vector orthogonal to the surface of the sphere, pointing outside
  gradf = np.array([2.0 * P0[0], 2.0 * P0[1], 2.0 * P0[2]])
  v = gradf / np.sqrt(np.dot(gradf, gradf))
  
  # East is ortogonal to the normal to the surface without Z component
  # that is a vector tangent to the circle at z = z0 (section of the sphere at z=z0)
  east = np.array([-2.0 * P0[1], 2.0 * P0[0], 0.0])  # orthogonal --> swap components, change sign of the first
  e = east / np.sqrt(np.dot(east, east))

  # North (n vector) is orthogonal to the Normal to the sphere (v - vertical vector) and to the East (e vector)
  north = np.cross(v, e)
  n = north / np.sqrt(np.dot(north, north))

  TransformMatrix = np.zeros((3,3))
  TransformMatrix[0, 0] = e[0]
  TransformMatrix[0, 1] = e[1]
  TransformMatrix[0, 2] = e[2]
  TransformMatrix[1, 0] = n[0]
  TransformMatrix[1, 1] = n[1]
  TransformMatrix[1, 2] = n[2]
  TransformMatrix[2, 0] = v[0]
  TransformMatrix[2, 1] = v[1]
  TransformMatrix[2, 2] = v[2]

  return TransformMatrix


if __name__ == "__main__":
  main()


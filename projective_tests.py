# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:43:18 2023

@author: pgbaz
"""

import numpy as np
import matplotlib.pyplot as plt

deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

class RotationVector():
  """
  https://developer.android.com/guide/topics/sensors/sensors_motion
  TYPE_ROTATION_VECTOR
  values[0]	Rotation vector component along the x axis (x * sin(θ/2))
  values[1]	Rotation vector component along the y axis (y * sin(θ/2))
  values[2]	Rotation vector component along the z axis (z * sin(θ/2))
  values[3]	Scalar component of the rotation vector ((cos(θ/2))
  The scalar component is an optional value. 
  Will NOT be used as it is not used in SkyMAP
  """
  def __init__(self):
    self.values = np.array([0.0, 0.0, 0.0, 0.0])


class GeoCoord():
  def __init__(self, lat, long, alt):
    self.latitude  = lat   # rad
    self.longitude = long  # rad
    self.altitude  = alt   # m


class Quaternion:
  """
  Quaternion in Android lib is composed by (please confirm !!!!):
  values[0]	w
  values[1]	ix
  values[2]	iy
  values[3]	iz
  Not so nice that it's an array: indexes can be confusing. Would be better to have a struct.
  Keeping as it is to comply with Android libs
  """
  def __init__(self):
    self.values = np.array([0.0, 0.0, 0.0, 0.0])

  def getQuaternionFromVector(self, rotationVector):
    """
    Get Quaternion from a 3 component rotation vector
    See https://android.googlesource.com/platform/frameworks/base/+/master/core/java/android/hardware/SensorManager.java
    See https://developer.android.com/reference/android/hardware/SensorManager#getQuaternionFromVector(float[],%20float[])
    The method getQuaternionFromVector (using code for rotation vector with 3 entries, consistent with SkyMAP)
    
    input: rotation vector with 3 entries
    output: initialize the quaternion instance "values" with the coefficients 
            of the quaternion corresponding to the provided rotation vector
    """
    q1 = rotationVector.values[0]
    q2 = rotationVector.values[1]
    q3 = rotationVector.values[2]
    q0 = 1.0 - q1 * q1 - q2 * q2 - q3 * q3
    
    if(q0 > 0.0):
      q0 = np.sqrt(q0)
    else:
      q0 = 0.0

    self.values[0] = q0
    self.values[1] = q1
    self.values[2] = q2
    self.values[3] = q3

      
  def invert(self):
    self.conjugate()
    res = Quaternion.multiply(self, self)
    mod2 = res[0]
    self.values[0] = self.values[0] / mod2
    self.values[1] = self.values[1] / mod2
    self.values[2] = self.values[2] / mod2
    self.values[3] = self.values[3] / mod2
    
  
  def conjugate(self):
    # values[0] i.e. W is unchanded when conjugating
    self.values[1] = -self.values[1]
    self.values[2] = -self.values[2]
    self.values[3] = -self.values[3]


  def copy(self, Q):
    """
    Copy the quaternion Q in the current instance by initializing the "values" from the input quaternion
    python alternative would be some deepcopy

    Parameters
    ----------
    Q : quaternion to be used for copying

    Returns
    -------
    None.
    """
    self.values[0] = Q.values[0]
    self.values[1] = Q.values[1]
    self.values[2] = Q.values[2]
    self.values[3] = Q.values[3]


  @staticmethod
  def multiply(Q0,Q1):
    """
    Multiplies two quaternions.
    NOTE: since quaternion multiply is non commutative, the order of the input parameters matters
     
    Input
    :param Q0: the first quaternion for the multiplication
    :param Q1: the second quaternion for the multiplication
     
    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33) 
    """
    
    w0 = Q0.values[0]
    x0 = Q0.values[1]
    y0 = Q0.values[2]
    z0 = Q0.values[3]
     
    w1 = Q1.values[0]
    x1 = Q1.values[1]
    y1 = Q1.values[2]
    z1 = Q1.values[3]
     
    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
     
    # Create a 4 element array containing the final quaternion
    # values[0] is the scalar, rest is the vector part x, y, z

    final_quaternion =Quaternion()
    final_quaternion.values[0] = Q0Q1_w
    final_quaternion.values[1] = Q0Q1_x
    final_quaternion.values[2] = Q0Q1_y
    final_quaternion.values[3] = Q0Q1_z

    return final_quaternion


def main():
  w = 1440.0
  h = 2418.0
  f = 1.0 # Focal length in mm, AoV=90

  sx =  w / 2.0
  sy = -h / 2.0
  tx =  w / 2.0
  ty =  h / 2.0

  # initialize the figure with the screen size (w, h)
  plt.figure()
  plt.xlim(-tx, tx)
  plt.ylim(-ty, ty)
  plt.gca().set_aspect('equal')

  # Building the rotation matrix
  phoneRotationMatrix = np.eye(4)
  phoneOrientation = getPhoneOrientation()
  phoneRotationMatrix = getRotationMatrixFromVector(phoneOrientation)
  
  # Build the rotation quaternion  
  phoneRotationQuaternion = Quaternion()
  phoneRotationQuaternionConj = Quaternion()
  phoneRotationQuaternion.getQuaternionFromVector(phoneOrientation)
  phoneRotationQuaternionConj.copy(phoneRotationQuaternion)
  phoneRotationQuaternionConj.conjugate()

  elev = 0.0 # elevation 0 is the line of the horizon
  elevRad = elev * np.pi / 180.0
  azRange = range(50, 131, 5)
  xVanish = np.zeros(np.size(azRange))
  yVanish = np.zeros(np.size(azRange))
  
  # phoneOrientation.values[0] = 0.021908345
  # phoneOrientation.values[1] = 0.009795881
  # phoneOrientation.values[2] = 0.58035123
  # phoneRotationQuaternion.getQuaternionFromVector(phoneOrientation)
  # print("Quat: ", phoneRotationQuaternion.values[0], phoneRotationQuaternion.values[1], phoneRotationQuaternion.values[2], phoneRotationQuaternion.values[3])
  # phoneRotationQuaternionConj.copy(phoneRotationQuaternion)
  # phoneRotationQuaternionConj.conjugate()
  # x1, y1, x1q, y1q, z1q, valid = transformPoint(3.0, 3.0, 0.0, phoneRotationQuaternion, phoneRotationQuaternionConj, sx, sy, tx, ty)
  # print(x1q, y1q, z1q)


  # compute points of the horizon
  idx = 0
  for az in azRange:
    azRad = float(az) * np.pi / 180.0
    x = np.cos(elevRad) * np.cos(azRad)
    y = np.cos(elevRad) * np.sin(azRad)
    z = np.sin(elevRad)

    #transform the vector with quaternion
    x1, y1, x1q, y1q, z1q, valid = transformPoint(x, y, z, phoneRotationQuaternion, phoneRotationQuaternionConj, sx, sy, tx, ty)
    if(valid):
      xVanish[idx] = x1
      yVanish[idx] = y1
      idx += 1

    # transform the vector with rotation matrix
    # enable this code to compare the results. Rotation matrix must be built outside of the loop
    # v = np.array([x, y, z, 1])
    # vt = phoneRotationMatrix.dot(v)
    # x1rm = vt[0]
    # y1rm = vt[1]
    # z1rm = vt[2]
    # print("x: ", x1rm-x1q, "y: ", y1rm-y1q, "z: ", z1rm-z1q)

  plt.plot(xVanish[0:idx], yVanish[0:idx], 'r')

  # An airplane at Aeroporto Urbe
  latitudePlane  = 41.95232550018577 * deg2rad
  longitudePlane = 12.505142833005202 * deg2rad
  altitudePlane = 3000.0
  planeGC = GeoCoord(latitudePlane, longitudePlane, altitudePlane)
  posPlane = getPointInEarthFrameFromGeoCoord(planeGC)
  
  # First obserer location: Settecamini
  latSettecamini  = 41.9401565969652 * deg2rad
  longSettecamini = 12.621029627805704 * deg2rad
  altSettecamini = 0.0  # altitude Settecamini = ?
  settecaminiGC = GeoCoord(latSettecamini, longSettecamini, altSettecamini)
  pos7c = getPointInEarthFrameFromGeoCoord(settecaminiGC)

  # Second observer location: Colosseo
  latColosseo  = 41.89014792072482 * deg2rad
  longColosseo = 12.492339876376782 * deg2rad
  altColosseo = 0.0  # altitude altColosseo = ?
  colosseoGC = GeoCoord(latColosseo, longColosseo, altColosseo)
  posColosseo = getPointInEarthFrameFromGeoCoord(colosseoGC)

  # ONLY FOR TEST: generate observation (az, el) from Observer 1 position
  observer1PlaneAz, observer1PlaneEl = generateObservation(settecaminiGC, planeGC, "Settecamini")

  # ONLY FOR TEST: generate observation (az, el) from Observer 2 position
  observer2PlaneAz, observer2PlaneEl = generateObservation(colosseoGC, planeGC, "Colosseo")

  compatible = checkAirplane(observer1PlaneAz, observer1PlaneEl, settecaminiGC, planeGC)
  print("compatible with Flighradar = ", compatible)

  compatible, P0 = checkUfo(observer1PlaneAz, observer1PlaneEl, settecaminiGC, observer2PlaneAz, observer2PlaneEl, colosseoGC)
  print("compatible with Ufo = ", compatible)

  xairplane = altitudePlane * np.cos(observer1PlaneEl) * np.cos(observer1PlaneAz)
  yairplane = altitudePlane * np.cos(observer1PlaneEl) * np.sin(observer1PlaneAz)
  zairplane = altitudePlane * np.sin(observer1PlaneEl)
  xpVanish, ypVanish, x1q, y1q, z1q, valid = transformPoint(xairplane, yairplane, zairplane, phoneRotationQuaternion, phoneRotationQuaternionConj, sx, sy, tx, ty)
  circle = plt.Circle((xpVanish, ypVanish), 20.0, color='b')
  plt.gca().add_patch(circle)
  plt.show()


def generateObservation(observerGC, planeGC, posName):
  # **************************************************************************************
  # BEGIN --> code only for test - Generates Az and El of an observation from observer 1
  # **************************************************************************************
  posObserver = getPointInEarthFrameFromGeoCoord(observerGC)
  posPlane = getPointInEarthFrameFromGeoCoord(planeGC)
  TransformMatrix = getTransformationMatrix(observerGC)
  posPlaneLoc1 = np.matmul(TransformMatrix, (posPlane - posObserver))
  observer1PlaneAz = np.arctan2(posPlaneLoc1[1], posPlaneLoc1[0])
  observer1PlaneEl = np.arctan2(posPlaneLoc1[2], np.sqrt(posPlaneLoc1[0] * posPlaneLoc1[0] + posPlaneLoc1[1] * posPlaneLoc1[1]))

  dist1 = pointPointDistance(posPlaneLoc1, np.array([0.0, 0.0, 0.0]))
  print("Position of the airplane in local ", posName, " frame = ", posPlaneLoc1)
  print("Dist = ", dist1, "Az = ", observer1PlaneAz * rad2deg, "El = ", observer1PlaneEl * rad2deg)
  # **************************************************************************************
  # END --> code only for test - Generates Az and El of an observation from observer 1
  # **************************************************************************************
  return observer1PlaneAz,observer1PlaneEl


def transformPoint(x, y, z, rotationQuaternion, rotationQuaternionConj, sx, sy, tx, ty):
  """
  Do the transformation by computing  qvq'
  Assumption not checked by the code:
    the rotation quaternion is well formed (unitary) so that it is not needed to
    compute the inverse of q but it is sufficient to use the conjugate.
  NOTE: xq1, yq1, zq1 are returned only for debug. No need of them in the production code

  """
  resultQuaternion = Quaternion()
  resultQuaternion.values[0] = 0.0
  resultQuaternion.values[1] = x
  resultQuaternion.values[2] = y
  resultQuaternion.values[3] = z
  resultQuaternion = Quaternion().multiply(rotationQuaternion, resultQuaternion)
  resultQuaternion = Quaternion().multiply(resultQuaternion, rotationQuaternionConj)
  xq1 = resultQuaternion.values[1]
  yq1 = resultQuaternion.values[2]
  zq1 = resultQuaternion.values[3]

  x1 = xq1
  y1 = zq1
  z1 = -yq1

  valid = False
  if(z1 < 0.0):
    valid = True

  x1 = (x1 / z1) * sx
  y1 = (y1 / z1) * sy

  return x1, y1, xq1, yq1, zq1, valid


def getPhoneOrientation():
  """
  Emulates the acquisition of the phone orientation from Android API

  Returns
  -------
  phoneOrientation : TYPE_ROTATION_VECTOR
    rotation vector with 3 entries composed by a normalized vector with 
    components(vx, vy, vz) defining the rotation axes, scaled by the 
    sin of the rotation angle. This construction justifies how the 
    getRotationMatrixFromVector and getQuaternionFromVector work.
    NOTE: when the angle is zero the rotation axe is irrelevant and the 
    returned rotation vector is zero
    TYPE_ROTATION_VECTOR:   [x*sin(alpha), y*sin(alpha), z*sin(alpha)[)]
  """
  phoneOrientation = RotationVector()
  
  # Axis definition
  vx = 0.0
  vy = 0.0
  vz = 1.0
  # Angle definition
  angle = -77.0
  
  if (angle == 0.0): # to avoid div by zero when axes and angle are all zero
    vx = 0.0
    vy = 0.0
    vz = 0.0
  else:
    # normalize axes andmultiply by sin angle
    modV = np.sqrt(vx*vx + vy*vy + vz*vz)
    st = np.sin(angle * np.pi / 180.0 / 2.0)
    vx = vx * st / modV
    vy = vy * st / modV
    vz = vz * st / modV
  
  phoneOrientation.values = np.array([vx, vy, vz])
  return phoneOrientation


def getRotationMatrixFromVector(rotationVector):
  """
  See: https://android.googlesource.com/platform/frameworks/base/+/master/core/java/android/hardware/SensorManager.java
  method: getRotationMatrixFromVector
  input: 
  output: the corresponding 4x4 rotation matrix

  Parameters
  ----------
  rotationVector: TYPE_ROTATION_VECTOR
    rotation vector with 3 entries

  Returns
  -------
  rotationMatrix : 4x4 rotation matrix
  """
  rotationMatrix = np.eye(4)  # declare identity as default rotation matrix 4x4
  q1 = rotationVector.values[0]
  q2 = rotationVector.values[1]
  q3 = rotationVector.values[2]
  q0 = 1.0 - q1 * q1 - q2 * q2 - q3 * q3
  
  if(q0 > 0.0):
    q0 = np.sqrt(q0)
  else:
    q0 = 0.0

  sq_q1 = 2.0 * q1 * q1
  sq_q2 = 2.0 * q2 * q2
  sq_q3 = 2.0 * q3 * q3
  q1_q2 = 2.0 * q1 * q2
  q3_q0 = 2.0 * q3 * q0
  q1_q3 = 2.0 * q1 * q3
  q2_q0 = 2.0 * q2 * q0
  q2_q3 = 2.0 * q2 * q3
  q1_q0 = 2.0 * q1 * q0

  rotationMatrix[0, 0] = 1.0 - sq_q2 - sq_q3
  rotationMatrix[0, 1] = q1_q2 - q3_q0
  rotationMatrix[0, 2] = q1_q3 + q2_q0
  rotationMatrix[0, 3] = 0.0
  rotationMatrix[1, 0] = q1_q2 + q3_q0
  rotationMatrix[1, 1] = 1 - sq_q1 - sq_q3
  rotationMatrix[1, 2] = q2_q3 - q1_q0
  rotationMatrix[1, 3] = 0.0
  rotationMatrix[2, 0] = q1_q3 - q2_q0
  rotationMatrix[2, 1] = q2_q3 + q1_q0
  rotationMatrix[2, 2] = 1.0 - sq_q1 - sq_q2
  rotationMatrix[2, 3] = 0.0
  rotationMatrix[3, 0] = 0.0
  rotationMatrix[3, 1] = 0.0 
  rotationMatrix[3, 2] = 0.0
  rotationMatrix[3, 3] = 1.0
  return rotationMatrix


def getPointInEarthFrameFromGeoCoord(GeoCoord):
  """
  Return the coordinates of the point in the earth frame given latitude and longitude

  Parameters
  ----------
  lat : latitude angle.  Unit: rad
  log : longitude angle. Unit: rad

  Returns
  -------
  x : coordinate x of the point in the earth frame. Unit: m
  y : coordinate y of the point in the earth frame. Unit: m
  z : coordinate z of the point in the earth frame. Unit: m

  """
  r = 6373044.737 # earth radius. Unit: meters
  r = r + GeoCoord.altitude
  x = r * np.cos(GeoCoord.latitude) * np.cos(GeoCoord.longitude)
  y = r * np.cos(GeoCoord.latitude) * np.sin(GeoCoord.longitude)
  z = r * np.sin(GeoCoord.latitude)
  P = np.array([x, y, z])
  return P


def pointPointDistance(P1, P2):
  q1 = (P1[0]-P2[0]) * (P1[0]-P2[0])
  q2 = (P1[1]-P2[1]) * (P1[1]-P2[1])
  q3 = (P1[2]-P2[2]) * (P1[2]-P2[2])
  d = np.sqrt(q1 + q2 + q3)
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
  
  b =     np.cos(el1) * np.cos(el2) * np.cos(az1) * np.cos(az2)
  b = b + np.cos(el1) * np.cos(el2) * np.sin(az1) * np.sin(az2)
  b = b + np.sin(el1) * np.sin(el2)
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

  k = x0 * np.cos(el) * np.cos(az) + y0 * np.cos(el) * np.sin(az) + z0 * np.sin(el)
  
  x1 = k * np.cos(el) * np.cos(az)
  y1 = k * np.cos(el) * np.sin(az)
  z1 = k * np.sin(el)
  P1 = np.array([x1, y1, z1])

  dist = pointPointDistance(P0, P1)
  dAux = pointPointDistance(P0, np.array([0.0, 0.0, 0.0]))
#  print("B) Coord point: ", dAux*np.cos(el)*np.cos(az), dAux*np.cos(el)*np.sin(az), dAux*np.sin(el))
#  print ("***", "\nP0 = ", P0[0], P0[1], P0[2], "\nP1 = ", P1[0], P1[1], P1[2], "\nDistance = ", dist, "\n***\n")
  
  return dist


def checkUfo(az1, el1, GeoCoordObserver1, az2, el2, GeoCoordObserver2):
  """
  Check whether 2 observations are compatible with each other.
  Supposed to run on cloud
 
  Parameters
  ----------
  az1 : float
    azimuth angle of UFO in the frame of observer 1.
  el1 : float
    elevation angle of UFO in the frame of observer 1.
  GeoCoordObserver1 : float array
    Contains latitude, longitude and altitude of observer 1 from GPS.
  az2 : float
    azimuth angle of UFO in the frame of observer 2.
  el2 : float
    elevation angle of UFO in the frame of observer 2.
  GeoCoordObserver2 : float array
    Contains latitude, longitude and altitude of observer 2 from GPS.

  Returns
  -------
  compatible : bool
    True if observation 1 and observation 2 are compatible.
  P0 : float array
    Coordinates of the Ufo in the frame of Observer1.
  """
  
  posObserver1 = getPointInEarthFrameFromGeoCoord(GeoCoordObserver1)
  posObserver2 = getPointInEarthFrameFromGeoCoord(GeoCoordObserver2)
  TransformMatrix = getTransformationMatrix(GeoCoordObserver1)
  posObs2InFrameObs1 = np.matmul(TransformMatrix, (posObserver2 - posObserver1))

  lineLineDist, P0, P1 = lineLineDistance(az1, el1, az2, el2, posObs2InFrameObs1)

  print("Pos Colosseum in frame 7C: ", posObs2InFrameObs1)
  print("Line-line dist: ", lineLineDist, "P0 = ", P0, "P1 = ", P1)

  # Let assume by now a linear increase of the error with the distance with increase factor equal to 1e-3
  ufoDist1 = pointPointDistance(P0, np.array([0.0, 0.0, 0.0]))
  ufoDist2 = pointPointDistance(P1, np.array([0.0, 0.0, 0.0]))
  ufoDist3 = pointPointDistance(P0, posObs2InFrameObs1)
  ufoDist4 = pointPointDistance(P1, posObs2InFrameObs1)

  ufoDist = ufoDist1
  if (ufoDist2 > ufoDist):
    ufoDist = ufoDist2
  if (ufoDist3 > ufoDist):
    ufoDist = ufoDist3
  if (ufoDist4 > ufoDist):
    ufoDist = ufoDist4

  print("ufoDist = ", ufoDist, "ufoDist1 = ", ufoDist1, "ufoDist2 = ", ufoDist2, "ufoDist3 = ", ufoDist3, "ufoDist4 = ", ufoDist4)

  coeff = 1e-3
  maxErr = coeff * ufoDist
  if(lineLineDist < maxErr):
    compatible = True
  else:
    compatible = False
  
  return compatible, P0


def checkAirplane(azAirplaneFromObs, elAirplaneFromObs, GeoCoordObserver, GeoCoordAirplane):
  """
  Check whether 1 observations is compatible with Flightradar.
  Supposed to run on the mobile (probably)
  Steps:  
    1) get local 3D coord of Airplane in the frame of the Observer
    2) compute the distance between observation and airplane (pointLineDistance)
    3) check if the observation is compatible with the Airplane
 
  Parameters
  ----------
  azAirplaneFromObs : float. Unit: rad
    azimuth angle of the airplane in the frame of the observer.
  elAirplaneFromObs : float. Unit: rad
    elevation angle of the airplane in the frame of the observer.
  GeoCoordObserver : class GPS()
    Contains latitude and longitude of observer 1 from GPS. Angles deg, altitude m
  GeoCoordAirplane : class GPS()
    Contains latitude, longitude and altitude from Flightradar. Angles deg, altitude m

  Returns
  -------
  compatible : bool
    True if observation 1 and observation 2 are compatible.

  """

  posObserver = getPointInEarthFrameFromGeoCoord(GeoCoordObserver)
  posPlane = getPointInEarthFrameFromGeoCoord(GeoCoordAirplane)
  TransformMatrix = getTransformationMatrix(GeoCoordObserver)

  posPlaneLoc = np.matmul(TransformMatrix, (posPlane - posObserver))
  planeAz = np.arctan2(posPlaneLoc[1], posPlaneLoc[0])
  planeEl = np.arctan2(posPlaneLoc[2], np.sqrt(posPlaneLoc[0] * posPlaneLoc[0] + posPlaneLoc[1] * posPlaneLoc[1]))

  ptLineDist = pointLineDistance(planeAz, planeEl, posPlaneLoc)

#  dist = pointPointDistance(posPlaneLoc, np.array([0.0, 0.0, 0.0]))
#  print("Position in local frame = ", posPlaneLoc)
#  print("Dist = ", dist, "Az = ", planeAz * rad2deg, "El = ", planeEl * rad2deg)
#  print("Point Line Distance = ", ptLineDist)

  # Let assume by now a linear increase of the error with the distance with increase factor equal to 1e-3
  airplaneDist = pointPointDistance(posPlaneLoc, np.array([0.0, 0.0, 0.0]))
  coeff = 1e-3
  maxErr = coeff * airplaneDist
  if(ptLineDist < maxErr):
    compatible = True
  else:
    compatible = False

  print("Observation to Airplane distance = ", ptLineDist)

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
  gradf = np.array([2.0*P0[0], 2.0*P0[1], 2.0*P0[2]])
  v = gradf / np.sqrt(np.dot(gradf, gradf))
  
  # East is ortogonal to the normal to the surface without Z component
  # that is a vector tangent to the circle at z = z0 (section of the sphere at z=z0)
  east = np.array([-2.0*P0[1], 2.0*P0[0], 0.0])  # orthogonal --> swap components, change sign of the first
  e = east / np.sqrt(np.dot(east, east))

  # North (n vector) is orthogonal to the Normal to the sphere (v - vertical vector) and to the East (e vector)
  n = np.cross(v, e) 

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

def getAngle(v1, v2):
  """
  After the generation of the transformation matrix the function is unused.
  To be removedsuring final cleanup in case it will still be unused.
  The function is anyway correct
  """
  angle = np.arccos(np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))))
  return angle


if __name__ == "__main__":
  main()


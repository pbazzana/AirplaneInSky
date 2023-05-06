# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:43:18 2023

@author: pgbaz
"""

import numpy as np
import matplotlib.pyplot as plt


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



  # Usato Settecamini per test
  latSettecamini  = 41.9401565969652 
  longSettecamini = 12.621029627805704
  altSettecamini = 0.0
  # altitude Settecamini = ?
  # abbreviazione x7c, y7c, z7c
  x7c, y7c, z7c = computePointInEarthFrame(latSettecamini, longSettecamini, altSettecamini)
  pos7c = np.array([x7c, y7c, z7c])
  
  # Usato aeroporto urbe per test con un aereo a una certa altitudine
  latitudePlane  = 41.95232550018577
  longitudePlane = 12.505142833005202
  altitudePlane = 500
  # abbreviazione xp, yp, zp
  xp, yp, zp = computePointInEarthFrame(latitudePlane, longitudePlane, altitudePlane)
  posPl = np.array([xp, yp, zp])
  
  xpg, ypg, zpg = computePointInEarthFrame(latitudePlane, longitudePlane, 0.0)
  posPlground = np.array([xpg, ypg, zpg])
  
  r5 = distance(pos7c, posPl)
  r3 = distance(pos7c, posPlground)
  planeEl = np.arccos(r3 / r5)

  d1 = r3
  x1, y1, z1 = computePointInEarthFrame(latSettecamini, longitudePlane, 0.0)
  deltay = y1 - y7c
  deltax = x1 - x7c
  alfa = np.arctan2(deltax, deltay)
  print(d1, alfa * 180 / np.pi, planeEl * 180 / np.pi)
  
  elevRad = planeEl
  azRad = alfa 
  altitude = altitudePlane
  
  xairplane = altitude * np.cos(elevRad) * np.cos(azRad)
  yairplane = altitude * np.cos(elevRad) * np.sin(azRad)
  zairplane = altitude * np.sin(elevRad)
  xpVanish, ypVanish, x1q, y1q, z1q, valid = transformPoint(xairplane, yairplane, zairplane, phoneRotationQuaternion, phoneRotationQuaternionConj, sx, sy, tx, ty)
  circle = plt.Circle((xpVanish, ypVanish), 20.0, color='b')
  plt.gca().add_patch(circle)
  plt.show()


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
  angle = -80.0
  
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


def computePointInEarthFrame(lat, lon, alt):
  """
  Return the coordinates of the point in the earth frame given latitude and longitude
  Note: by now neglect altitude information. 
  The observers are assumed to be at the same altitude.

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
  la = lat * np.pi /180.0
  lo = lon * np.pi /180.0
  r = 6373044.737 # earth radius. Unit: meters
  x = r * np.cos(la) * np.cos(lo)
  y = r * np.cos(la) * np.sin(lo)
  z = r * np.sin(la) + alt
  return x, y, z


def distance(P1, P2):
  d = np.sqrt((P1[0]-P2[0])*(P1[0]-P2[0]) + (P1[1]-P2[1])*(P1[1]-P2[1]) + (P1[2]-P2[2])*(P1[2]-P2[2]))
  return d


if __name__ == "__main__":
  main()


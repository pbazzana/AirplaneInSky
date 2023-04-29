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
    Get Quaternion from a 3 component vector
    See https://android.googlesource.com/platform/frameworks/base/+/master/core/java/android/hardware/SensorManager.java
    The method getQuaternionFromVector (using code for rotation vector with 3 entries, consistent with SkyMAP)
    
    input: rotation vector with 3 entries
    output: initialize the quaternion instance "values" with the coefficients 
            of the quaternion corresponding to the provided rotation vector
    """
    rv = rotationVector
    self.values[0] = 1.0 - rv.values[0] * rv.values[0] - rv.values[1] * rv.values[1] - rv.values[2] * rv.values[2]
    if(self.values[0] > 0):
      self.values[0] = np.sqrt(self.values[0])
    else:
      self.values[0] = 0.0
    self.values[1] = rv.values[0]
    self.values[2] = rv.values[1]
    self.values[3] = rv.values[2]
  
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
    # original code of Android: suggests that values[0] is W (the scalar of the quaternion)
    # np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])
    
    final_quaternion =Quaternion()
    final_quaternion.values[0] = Q0Q1_w
    final_quaternion.values[1] = Q0Q1_x
    final_quaternion.values[2] = Q0Q1_y
    final_quaternion.values[3] = Q0Q1_z

    return final_quaternion
    

def main():
  elev = 0.0 # elevation 0 is the line of the horizon
  elevRad = elev * np.pi / 180.0
  r = range(-45, 46, 5)
  xVanish = np.zeros(np.size(r))
  yVanish = np.zeros(np.size(r))

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
  resultQuaternion = Quaternion()
  
  idx = 0
  for az in r:
    azRad = az * np.pi / 180
    z = np.cos(elevRad) * np.cos(azRad)
    x = np.cos(elevRad) * np.sin(azRad)
    y = np.sin(elevRad)
    
    #transform the vector with matrix
    v = np.array([x, y, z, 1])
    vt = phoneRotationMatrix.dot(v)
    x1 = vt[0]
    y1 = vt[1]
    z1 = vt[2]

    #transform the vector with quaternion
    resultQuaternion.values[0] = 0.0
    resultQuaternion.values[1] = x
    resultQuaternion.values[2] = y
    resultQuaternion.values[3] = z
    resultQuaternion = Quaternion().multiply(phoneRotationQuaternion, resultQuaternion)
    resultQuaternion = Quaternion().multiply(resultQuaternion, phoneRotationQuaternionConj)
    xq1 = resultQuaternion.values[1]
    yq1 = resultQuaternion.values[2]
    zq1 = resultQuaternion.values[3]
    
    if (np.abs(z1) > 0.0):
      xVanish[idx] = x1 / z1
      yVanish[idx] = y1 / z1
      
    idx += 1

  plt.figure()
  plt.plot(xVanish, yVanish)
  plt.show()


def getPhoneOrientation():
  """
  Emulates the acquisition of the phone orientation from Android API

  Returns
  -------
  phoneOrientation : TYPE_ROTATION_VECTOR
    rotation vector with 3 entries
  """
  phoneOrientation = RotationVector()
  st = np.sin(-25.0*np.pi/180.0)
  phoneOrientation.values = np.array([0.0, st, 0.0])
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


if __name__ == "__main__":
  main()


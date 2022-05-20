import numpy as np
import scipy
from scipy import linalg
import control
import math

def k_gain_calculator(Q_state,R_state):
    # defined dynamics parameters
    mb = 4.0
    mB = 3.3155
    mw = 1.098
    # mB = 4.9064
    # mw = 1.415
    rb = 0.12
    rw = 0.05
    l  = 0.208
    Ib = 0.01165
    IB = 0.3
    Iw = 0.00004566491
    g  = -9.81
    IB_xy = 0.3
    Iw_xy = 0.0002770398
    ###############
    A_mat_3_2 = -(g*(rb**4)*(l*mB + mw*rb + mw*rw)*( (mw*(rw**3)) - Iw*rb + (l*mB*(rw**2)) + (mw*rb*(rw**2)))) / ( (IB*Ib*(rw**2)) + (IB*Iw*(rb**2)) + (Ib*Iw*(rb**2)) + (Iw*mB*(rb**4)) + (Iw*mb*(rb**4)) + (Ib*mw*(rw**4)) + (4*Iw*mw*(rb**4)) + (2*Iw*l*mB*(rb**3)) + (2*Ib*mw*rb*(rw**3)) + (4*Iw*mw*(rb**3)*rw) - ((l**2)*(mB**2)*(rb**2)*(rw**2)) + (IB*mB*(rb**2)*(rw**2)) + (IB*mb*(rb**2)*(rw**2)) + (IB*mw*(rb**2)*(rw**2)) + (Ib*mw*(rb**2)*(rw**2)) + (Iw*mw*(rb**2)*(rw**2)) + (mB*mw*(rb**2)*(rw**4)) + (2*mB*mw*(rb**3)*(rw**3)) + (mB*mw*(rb**4)*(rw**2)) + (mb*mw*(rb**2)*(rw**4)) + (2*mb*mw*(rb**3)*(rw**3)) + (mb*mw*(rb**4)*(rw**2)) - (2*l*mB*mw*(rb**2)*(rw**3)) - (2*l*mB*mw*(rb**3)*(rw**2)))
    A_mat_4_2 = (g*(l*mB + mw*rb + mw*rw)*( (Ib*(rw**2)) + (Iw*(rb**2)) + (mB*(rb**2)*(rw**2)) + (mb*(rb**2)*(rw**2)) + (mw*(rb**2)*(rw**2))))/( (IB*Ib*(rw**2)) + (IB*Iw*(rb**2)) + (Ib*Iw*(rb**2)) + (Iw*mB*(rb**4)) + (Iw*mb*(rb**4)) + (Ib*mw*(rw**4)) + (4*Iw*mw*(rb**4)) + (2*Iw*l*mB*(rb**3)) + (2*Ib*mw*rb*(rw**3)) + (4*Iw*mw*(rb**3)*rw) - ((l**2)*(mB**2)*(rb**2)*(rw**2)) + (IB*mB*(rb**2)*(rw**2)) + (IB*mb*(rb**2)*(rw**2)) + (IB*mw*(rb**2)*(rw**2)) + (Ib*mw*(rb**2)*(rw**2)) + (Iw*mw*(rb**2)*(rw**2)) + (mB*mw*(rb**2)*(rw**4)) + (2*mB*mw*(rb**3)*(rw**3)) + (mB*mw*(rb**4)*(rw**2)) + (mb*mw*(rb**2)*(rw**4)) + (2*mb*mw*(rb**3)*(rw**3)) + (mb*mw*(rb**4)*(rw**2)) - (2*l*mB*mw*(rb**2)*(rw**3)) - (2*l*mB*mw*(rb**3)*(rw**2)))
    B_mat_3 = ((rb**2)*rw*( (2*mw*(rb**2)) + (3*mw*rb*rw) + (l*mB*rb) + (mw*rw**2) + IB))/( (IB*Ib*(rw**2)) + (IB*Iw*(rb**2)) + (Ib*Iw*(rb**2)) + (Iw*mB*(rb**4)) + (Iw*mb*(rb**4)) + (Ib*mw*(rw**4)) + (4*Iw*mw*(rb**4)) + (2*Iw*l*mB*(rb**3)) + (2*Ib*mw*rb*(rw**3)) + (4*Iw*mw*(rb**3)*rw) - ((l**2)*(mB**2)*(rb**2)*(rw**2)) + (IB*mB*(rb**2)*(rw**2)) + (IB*mb*(rb**2)*(rw**2)) + (IB*mw*(rb**2)*(rw**2)) + (Ib*mw*(rb**2)*(rw**2)) + (Iw*mw*(rb**2)*(rw**2)) + (mB*mw*(rb**2)*(rw**4)) + (2*mB*mw*(rb**3)*(rw**3)) + (mB*mw*(rb**4)*(rw**2)) + (mb*mw*rb**2*rw**4) + (2*mb*mw*(rb**3)*(rw**3)) + (mb*mw*(rb**4)*(rw**2)) - (2*l*mB*mw*(rb**2)*(rw**3)) - (2*l*mB*mw*(rb**3)*(rw**2)))

    B_mat_4 = -(rb*rw*(Ib + (mB*(rb**2)) + (mb*(rb**2)) + (2*mw*rb**2) + (l*mB*rb) + (mw*rb*rw)))/((IB*Ib*(rw**2)) + (IB*Iw*(rb**2)) + (Ib*Iw*(rb**2)) + (Iw*mB*(rb**4)) + (Iw*mb*(rb**4)) + (Ib*mw*(rw**4)) + (4*Iw*mw*(rb**4)) + (2*Iw*l*mB*(rb**3)) + 2*Ib*mw*rb*rw**3 + 4*Iw*mw*rb**3*rw - l**2*mB**2*rb**2*rw**2 + IB*mB*rb**2*rw**2 + IB*mb*rb**2*rw**2 + IB*mw*rb**2*rw**2 + Ib*mw*rb**2*rw**2 + Iw*mw*rb**2*rw**2 + mB*mw*rb**2*rw**4 + 2*mB*mw*rb**3*rw**3 + mB*mw*rb**4*rw**2 + mb*mw*rb**2*rw**4 + 2*mb*mw*rb**3*rw**3 + mb*mw*rb**4*rw**2 - 2*l*mB*mw*rb**2*rw**3 - 2*l*mB*mw*rb**3*rw**2)
    B_mat_xy = (-rb/( (IB_xy*rw**2) + (mw*rw**2*(rb+rw)**2 + (Iw_xy*rb**2))))
    A_sys = np.array([ [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, A_mat_3_2, 0, 0, 0, 0, 0, 0], 
                    [A_mat_3_2, 0, 0, 0, 0, 0, 0, 0],
                    [0, A_mat_4_2, 0, 0, 0, 0, 0, 0],
                    [A_mat_4_2, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0] ])
    B_sys =np.array([ [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [B_mat_3, 0, 0],
                    [0, B_mat_3, 0],
                    [B_mat_4, 0, 0],
                    [0, B_mat_4, 0],
                    [0, 0, B_mat_xy]])
    C_sys = np.eye(8)
    D_sys = np.zeros((8,3))
    # Q_state = np.array([5, 5, 5, 30, 30, 30, 30, 30])
    # R_state = np.array([10, 10, 10])
    Q = np.eye(8) * Q_state
    R = np.eye(3) * R_state
    K, S, E = control.lqr(A_sys, B_sys, Q, R)
    # print("A_sys =", A_sys)
    # print("B_sys =", B_sys)
    # print("C_sys =", C_sys)
    # print("D_sys =", D_sys)
    # print("K =", K)
    # print(2*5**2*3*2)
    return K
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return [roll_x, pitch_y, yaw_z] # in radians

def lqr_controller(x_ref,x_fb,K):
    u= K @ (x_ref-x_fb)
    return u
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return [roll_x, pitch_y, yaw_z] # in radians

def ball_velocity(pre_vel,now_vel,dt):
    v = (now_vel-pre_vel)/dt
    return v 

def Txyz2wheel(Txyz):
    alpha = 50.0*np.pi/180.0 # wheelaxis_theta 
    Twheels = np.array([[-np.cos(alpha)  ,              0            ,       -np.sin(alpha)],
                        [np.cos(alpha)/2   ,  -np.sqrt(3)*np.cos(alpha)/2 ,   -np.sin(alpha)],
                        [np.cos(alpha)/2   ,  np.sqrt(3)*np.cos(alpha)/2,    -np.sin(alpha)]]) @ Txyz # -x axis
    # Twheels = np.array([[ np.cos(alpha) ,              0            ,          -np.sin(alpha)],
    #                     [-np.cos(alpha)/2   ,   np.sqrt(3)*np.cos(alpha)/2 ,   -np.sin(alpha)],
    #                     [-np.cos(alpha)/2   ,  -np.sqrt(3)*np.cos(alpha)/2,    -np.sin(alpha)]]) @ Txyz  #  x axis
    return Twheels
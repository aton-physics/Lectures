#!/usr/bin/python
from scipy.integrate import odeint
import matplotlib.pyplot as plt # for plotting  
import math
import numpy as np
from copy import copy
from mpl_toolkits.mplot3d import Axes3D

class Particle3D(object):

    """Class that describes particle"""
    m = 1.0

    def __init__(self, x0=0.0, v_x0=0.0,  y0=0.0, v_y0 = 0.0, z0 = 100.0, v_z0 = 0.0, tf = 10.0, dt = 0.01, tag = "tag", omegax = 0.0, omegay = 0.0, omegaz = 0.0):
        self.x = x0
        self.v_x = v_x0
        self.y = y0
        self.v_y = v_y0
        self.z = z0
        self.v_z = v_z0
        self.t = 0.0
        self.tf = tf
        self.dt = dt
        self.counter = 0
        
        self.omegax = omegax
        self.omegay = omegay
        self.omegaz = omegaz

        self.tlabel = 'time (s)'
        self.xlabel = 'x (m)'
        self.ylabel = 'y (m)'
        self.zlabel = 'z (m)'
        self.v_xlabel = 'v_x (m/s)'
        self.v_ylabel = 'v_y (m/s)'
        self.v_yzlabel = 'v_z (m/s)'
        self.tag = tag
        self.initial_potential = -self.m * 9.81 * z0 + 0.5*self.m*(v_x0**2 + v_y0**2 + v_z0**2)
        self.current_energy = []
        self.drag_work = []
        self.hamiltonian = []
        self.drag_accum = 0.0

        npoints = int(tf/dt) # always starting at t = 0.0
        self.npoints = npoints
        self.tarray = np.linspace(0.0, tf, npoints, endpoint = True) # include final timepoint
        self.xyv_xv_y = np.array([self.x, self.y, self.z, self.v_x, self.v_y, self.v_z]) # NumPy array with initial position and velocity
        self.rx = []
        self.ry = []
        self.rz = []
        self.rvx = []
        self.rvy = []
        self.rvz = []

        print("A new particle has been init'd")

    def drag_coefficient(self, v, r=1):
        a = 0.25
        b = 0.25
        chi = (abs(v)-10.0)/4.0
        #print(chi)
        term1 = a + b/(1.0+np.exp(chi))
        if (chi <= 0):
            term2 = np.exp(-(chi**2))
        elif (chi > 0):
            term2 = np.exp(-(chi)**2 / 4)
        else:
            print("bad variable")
        term2 *= -0.16
        return (term1+term2)*3.14159*r**2
    
    def magnus_force(self, v_x, v_y, v_z, r=0.25):
        #return lift forces not yet multiplied by rho
        vel_sq = v_x**2 + v_y**2 + v_z**2
        vel = math.sqrt(vel_sq)
        omegax = self.omegax*np.exp(-self.t)
        omegay = self.omegay*np.exp(-self.t)
        omegaz = self.omegaz*np.exp(-self.t)
        omega = math.sqrt(omegax**2 + omegay**2 + omegaz**2)
        if (vel == 0):
            print("zero velocity")
        spin = r*omega/vel
        if (spin < 0.00001):
            lift_force = [0,0,0]
            return lift_force
        lift_constant = 0.5*spin**(0.4)
        omega_cross_v = [omegay*v_z - omegaz*v_y, omegaz*v_x - omegax*v_z , omegax*v_y - omegay*v_x]
        lift_force = [0.5*lift_constant*r/spin*omega_cross_v[0], 0.5*lift_constant*r/spin*omega_cross_v[1], 0.5*lift_constant*r/spin*omega_cross_v[2]]
        return lift_force
    
    def force(self, x, y, z, v_x, v_y, v_z, optional = 0, rk_step = 0):
        dt = self.dt
        rho = 1.225 #units = kg / m^3
        vel_sq = v_x**2 + v_y**2 + v_z**2
        vel = math.sqrt(vel_sq)
        drag_coeff = [self.drag_coefficient(v_x), self.drag_coefficient(v_y), self.drag_coefficient(v_z)]
        force = [-0.5 * drag_coeff[0]* rho * vel * v_x, -0.5 * drag_coeff[1] * 
                 rho * vel * v_y, -0.5*drag_coeff[2] * rho * vel * v_z - 9.81]
        #work_from_drag = abs(force[0]*(v_x*dt)) + abs(force[1]*(v_y*dt)) + abs((force[2]+9.81)*(v_z*dt))
        work_from_drag = -force[0]*(v_x*dt) - force[1]*(v_y*dt) - (force[2]+9.81)*(v_z*dt)
        #print(work_from_drag)
        lift_force = self.magnus_force(v_x, v_y, v_z)
        for ii in range(0,3):
            lift_force[ii] *= rho
            force[ii] += lift_force[ii]
        #work_from_drag = -force[0]*(self.v_x*dt) - force[1]*(self.v_y*dt) - (force[2]+9.81)*(self.v_z*dt)    
        if (optional == 0):
            if (rk_step == 1 or rk_step == 4):
                self.drag_accum += work_from_drag / 6
            elif (rk_step == 2 or rk_step == 3):
                self.drag_accum += work_from_drag / 3
            else:
                print("Something's wrong.")
        if (optional == 1):
            self.drag_work.append(self.drag_accum)
            self.current_energy.append(0.5*self.m*vel_sq - self.m*9.81*z)
            self.hamiltonian.append(0.5*self.m*vel_sq - self.m*9.81*z - self.drag_accum)
        return force
    
    def RK4_step(self):
        """
        Take a single time step using RK4 midpoint method
        """
        first = self.force(self.x, self.y, self.z, self.v_x, self.v_y, self.v_z, 0, 1)
        
        a1 = first[0]/ self.m
        b1 = first[1]/ self.m
        c1 = first[2]/ self.m
        k1 = np.array([self.v_x, a1])*self.dt
        l1 = np.array([self.v_y, b1])*self.dt
        m1 = np.array([self.v_z, c1])*self.dt
        
        second = self.force(self.x + k1[0]/2, self.y + l1[0]/2, self.z + m1[0]/2, self.v_x + k1[1]/2, self.v_y + l1[1]/2, self.v_z + m1[1]/2, 0, 2)
        
        a2 = second[0] / self.m
        b2 = second[1] / self.m
        c2 = second[2] / self.m
        k2 = np.array([self.v_x + k1[1]/2, a2])*self.dt
        l2 = np.array([self.v_y + l1[1]/2, b2])*self.dt
        m2 = np.array([self.v_z + m1[1]/2, c2])*self.dt
        
        third = self.force(self.x + k2[0]/2, self.y + l2[0]/2, self.z + m2[0]/2, self.v_x + k2[1]/2, self.v_y + l2[1]/2, self.v_z + m2[1]/2, 0, 3)
        
        a3 = third[0] / self.m
        b3 = third[1] / self.m
        c3 = third[2] / self.m
        k3 = np.array([self.v_x + k2[1]/2, a3])*self.dt
        l3 = np.array([self.v_y + l2[1]/2, b3])*self.dt
        m3 = np.array([self.v_z + m2[1]/2, c3])*self.dt
        
        fourth = self.force(self.x + k3[0], self.y + l3[0], self.z + m3[0], self.v_x + k3[1], self.v_y + l3[1], self.v_z + m3[1], 0, 4)
        
        a4 = fourth[0] / self.m
        b4 = fourth[1] / self.m
        c4 = fourth[2] / self.m
        k4 = np.array([self.v_x + k3[1], a4])*self.dt
        l4 = np.array([self.v_y + l3[1], b4])*self.dt
        m4 = np.array([self.v_z + m3[1], c4])*self.dt

        self.x += (k1[0]+ k4[0])/6 + (k2[0] + k3[0])/3
        self.y += (l1[0]+ l4[0])/6 + (l2[0] + l3[0])/3
        self.z += (m1[0]+ m4[0])/6 + (m2[0] + m3[0])/3
        self.v_x += (k1[1]+ k4[1])/6 + (k2[1] + k3[1])/3
        self.v_y += (l1[1]+ l4[1])/6 + (l2[1] + l3[1])/3
        self.v_z += (m1[1]+ m4[1])/6 + (m2[1] + m3[1])/3
        
        self.t += self.dt
        
    def RK4_trajectory(self): 
        """
        Loop over all time steps to construct a trajectory with RK4 method
        Will reinitialize euler trajectory everytime this method is called
        """
        
        x_RK4 = []
        y_RK4 = []
        z_RK4 = []
        v_xRK4 = []
        v_yRK4 = []
        v_zRK4 = []
        
        while(self.t < self.tf - self.dt/2):
            first = self.force(self.x, self.y, self.z, self.v_x, self.v_y, self.v_z, 1)
            x_RK4.append(self.x)
            y_RK4.append(self.y)
            z_RK4.append(self.z)
            v_xRK4.append(self.v_x)
            v_yRK4.append(self.v_y)
            v_zRK4.append(self.v_z)
            self.RK4_step()
        
        self.x= np.array(x_RK4)
        self.y= np.array(y_RK4)
        self.z= np.array(z_RK4)
        self.v_x= np.array(v_xRK4)
        self.v_y= np.array(v_yRK4)
        self.v_z= np.array(v_zRK4)
        
    def RK4_plot(self):
        fig1=plt.figure()
        plt.ticklabel_format(useOffset=False)
        ax1 = plt.axes(projection='3d')
        ax1.plot3D(self.x, self.y, self.z, "k", label = 'RK4')
        ax1.set_title(self.tag)
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_zlabel('z (m)');
        
        #ax1.plot(self.tarray, self.x_RK4, "r", label = 'RK4')
        #ax2.plot(self.x_RK4, self.v_RK4, "r", label = 'RK4')
        
    def energy_conservation(self):
        fig1=plt.figure()
        ax1 = fig1.add_subplot(111)
        plt.ticklabel_format(useOffset=False)
        #time = np.linspace(0.0, 10, len(self.drag_work), endpoint = True)
        ax1.plot(self.tarray, self.drag_work, "k--", label = 'W_drag')
        ax1.set_title(self.tag)
        ax1.set_xlabel("time (sec)")
        ax1.set_ylabel("Energy (J)")
        
        ax1.plot(self.tarray, self.current_energy, "k:", label = "E(t)")
        ax1.plot(self.tarray, self.hamiltonian, "k", label = "E(t) - W_drag")
        legend = ax1.legend(loc='best', shadow=True, fontsize='x-large')
        
        #ax1.plot(self.tarray, self.xv[:, 0], "k", label = 'odeint')
        #ax2.plot(self.xv[:, 0], self.xv[:, 1], "k", label = 'odeint')
    #def energy_conservation(self):
        #print(len(self.drag_work), len(self.current_energy))


    def scipy_trajectory(self, optional = 0):
        """calculate trajectory using SciPy ode integrator"""
        if (optional == 0):
            soln = odeint(self.derivative, self.xyv_xv_y, self.tarray)
        else:
            print("bad argument")
        self.rx = soln[:,0]
        self.ry = soln[:,1]
        self.rz = soln[:,2]
        self.rvx = soln[:,3]
        self.rvy = soln[:,4]
        self.rvz = soln[:,5]
        
    def derivative(self, xv, t):
        self.counter += 1
        print("derivative called" + str(self.counter) + " times.")
        """right hand side of the differential equation
            Required for odeint """
        x =xv[0]
        y =xv[1]
        z = xv[2]
        vx =xv[3]
        vy = xv[4]
        vz = xv[5]
        F = self.force(x,y,z,vx,vy,vz)
        ax = F[0]/self.m
        ay = F[1]/self.m
        az = F[2]/self.m
        #print(ax)
        #a = self.F(x, v, t) / self.m
        #return np.ravel(np.array([v, a]))
        return [vx, vy, vz, ax, ay, az]
    
    def plot(self):
        fig1=plt.figure()
        plt.ticklabel_format(useOffset=False)
        ax1 = plt.axes(projection='3d')
        ax1.plot3D(self.rx, self.ry, self.rz, "k", label = 'odeint')
        ax1.set_title(self.tag)
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_zlabel('z (m)');
        
    def test(self):
        for ii in self.hamiltonian:
            print(ii)

class Particle(object):

    """Class that describes particle"""
    m = 1.0

    def __init__(self, x0=0.0, v_x0=0.0,  y0=0.0, v_y0 = 0.0, tf = 10.0, dt = 0.001, tag = "tag"):
        self.x = x0
        self.v_x = v_x0
        self.y = y0
        self.v_y = v_y0
        self.t = 0.0
        self.tf = tf
        self.dt = dt

        self.tlabel = 'time (s)'
        self.xlabel = 'x (m)'
        self.ylabel = 'y (m)'
        self.v_xlabel = 'v_x (m/s)'
        self.v_ylabel = 'v_y (m/s)'
        self.tag = tag

        npoints = int(tf/dt) # always starting at t = 0.0
        self.npoints = npoints
        self.tarray = np.linspace(0.0, tf,npoints, endpoint = True) # include final timepoint
        self.xyv_xv_y = np.array([self.x, self.y, self.v_x, self.v_y]) # NumPy array with initial position and velocity
        self.rx = []
        self.ry = []
        self.rvx = []
        self.rvy = []

        print("A new particle has been init'd")

    def F(self, x, v, t):
        # The force on a free particle is 0
        
        return array([0.0])
    def density(self, y):
        T = 300
        a = 6.5*10**-3
        variable = a*y/T
        expo = 2.5
        return 1.225*(1.0 - variable)**2.5
    
    def drag_force(self, x, y, v_x, v_y):
        rho = self.density(y)
        vel_sq = v_x**2 + v_y**2
        vel = math.sqrt(vel_sq)
        g = 6.67*10**-11 *5.972*10**24 / ((6371000 + y)**2)
        force = [1 / 4 * rho * vel * v_x, 1 / 4 * rho * vel * v_y + g]
        return force
    
    def drag_force_const_density(self, x, y, v_x, v_y):
        rho = self.density(0)
        vel_sq = v_x**2 + v_y**2
        vel = math.sqrt(vel_sq)
        g = 6.67*10**-11 *5.972*10**24 / ((6371000 + y)**2)
        force = [1 / 4 * rho * vel * v_x, 1 / 4 * rho * vel * v_y + g]
        return force
    
    def Euler_step(self): 
        """
        Take a single time step using Euler method
        """
        
        a = self.F(self.x, self.v, self.t) / self.m
        self.x += self.v * self.dt
        self.v += a * self.dt
        self.t += self.dt

    def RK4_step(self):
        """
        Take a single time step using RK4 midpoint method
        """
        a1 = self.F(self.x, self.v, self.t) / self.m
        k1 = np.array([self.v, a1])*self.dt

        a2 = self.F(self.x+k1[0]/2, self.v+k1[1]/2, self.t+self.dt/2) / self.m
        k2 = np.array([self.v+k1[1]/2 ,a2])*self.dt
        
        a3 = self.F(self.x+k2[0]/2, self.v+k2[1]/2, self.t+self.dt/2) / self.m
        k3 = np.array([self.v+k2[1]/2, a3])*self.dt
        
        a4 = self.F(self.x+k3[0], self.v+k3[1], self.t+self.dt) / self.m
        k4 = np.array([self.v+k3[1], a4])*self.dt

        self.x += (k1[0]+ k4[0])/6 + (k2[0] + k3[0])/3
        self.v += (k1[1]+ k4[1])/6 + (k2[1] + k3[1])/3
        
        self.t += self.dt

    def Euler_trajectory(self):  
        """
        Loop over all time steps to construct a trajectory with Euler method
        Will reinitialize euler trajectory everytime this method is called
        """
        
        x_euler = []
        v_euler = []
        
        while(self.t < self.tf-self.dt/2):
            v_euler.append(self.v)
            x_euler.append(self.x)
            self.Euler_step()
        
        self.x_euler = np.array(x_euler)
        self.v_euler = np.array(v_euler)
    
    def RK4_trajectory(self): 
        """
        Loop over all time steps to construct a trajectory with RK4 method
        Will reinitialize euler trajectory everytime this method is called
        """
        
        x_RK4 = []
        v_RK4 = []
        
        while(self.t < self.tf - self.dt/2):
            x_RK4.append(self.x)
            v_RK4.append(self.v)
            self.RK4_step()

        self.x_RK4 = np.array(x_RK4)
        self.v_RK4 = np.array(v_RK4)

    def scipy_trajectory(self, optional = 0):
        """calculate trajectory using SciPy ode integrator"""
        if (optional == 0):
            soln = odeint(self.derivative, self.xyv_xv_y, self.tarray)
        elif (optional == 1):
            soln = odeint(self.rhs_constant_density, self.xyv_xv_y, self.tarray)
        elif (optional == 2):
            soln = odeint(self.rhs_free, self.xyv_xv_y, self.tarray)
        self.rx = soln[:,0]
        self.ry = soln[:,1]
        self.rvx = soln[:,2]
        self.rvy = soln[:,3]
    def derivative(self, xv, t):
        """right hand side of the differential equation
            Required for odeint """
        x =xv[0]
        y =xv[1]
        vx =xv[2]
        vy = xv[3]
        F = self.drag_force(x,y,vx,vy)
        ax = -F[0]/self.m
        ay = -F[1]/self.m
        #a = self.F(x, v, t) / self.m
        #return np.ravel(np.array([v, a]))
        return [vx, vy, ax, ay]
    
    def rhs_constant_density(self, xv, t):
        """right hand side of the differential equation
            Required for odeint """
        x =xv[0]
        y =xv[1]
        vx =xv[2]
        vy = xv[3]
        F = self.drag_force_const_density(x,y,vx,vy)
        ax = -F[0]/self.m
        ay = -F[1]/self.m
        #a = self.F(x, v, t) / self.m
        #return np.ravel(np.array([v, a]))
        return [vx, vy, ax, ay]
    
    def rhs_free(self, xv, t):
        """right hand side of the differential equation
            Required for odeint """
        x =xv[0]
        y =xv[1]
        vx =xv[2]
        vy = xv[3]
        g = 6.67*10**-11 *5.972*10**24 / ((6371000 + y)**2)
        F = [0, g]
        ax = -F[0]/self.m
        ay = -F[1]/self.m
        #a = self.F(x, v, t) / self.m
        #return np.ravel(np.array([v, a]))
        return [vx, vy, ax, ay]
    
    def results(self):
        """ 
        Print out results in a nice format
        """
        
        print('\n\t Position and Velocity at Final Time:')
        print('Euler:')
        print('t = {} x = {} v = {}'.format(self.t, self.x , self.v))
        
        if hasattr(self, 'xv'):
            print('SciPy ODE Integrator:')
            print('t = {} x = {} v = {}'.format(self.tarray[-1], self.xv[-1, 0], self.xv[-1,1]))

    def plotkemp(self):
        """ 
        Make nice plots of our results
        """

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        
        if hasattr(self,'xv'):
            ax1.plot(self.tarray, self.xv[:, 0], "k", label = 'odeint')
            ax2.plot(self.xv[:, 0], self.xv[:, 1], "k", label = 'odeint')
        if hasattr(self,'x_euler'):
            ax1.plot(self.tarray, self.x_euler, "b", label = 'euler')
            ax2.plot(self.x_euler, self.v_euler, "b", label = 'euler')
        if hasattr(self,'x_RK4'):
            ax1.plot(self.tarray, self.x_RK4, "r", label = 'RK4')
            ax2.plot(self.x_RK4, self.v_RK4, "r", label = 'RK4')
            
        ax1.set_title('Trajectory')
        ax1.set_xlabel("t")
        ax1.set_ylabel("x")
        
        ax2.set_title('Phase space')
        ax2.set_xlabel("v")
        ax2.set_ylabel("x")

        ax1.legend()
        ax2.legend()
    def plot(self):
        fig1=plt.figure()
        plt.ticklabel_format(useOffset=False)
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.rx[:], self.ry[:], "k", label = 'odeint')
        ax1.set_title(self.tag)
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")

class FallingParticle(Particle):

    """Subclass of Particle Class that describes a falling particle"""
    g = 9.8

    def __init__(self,m = 1.0, x0 = 1.0 , v0 = 0.0, tf = 10.0,  dt = 0.1):
        self.m = m
        super().__init__(x0,v0,tf,dt)   # call initialization method of the super (parent) class
    
    def F(self, x, v, t):
            return  -self.g*self.m

class ChargedParticle(Particle3D):
    
    def __init__(self, x0=0.0, v_x0=0.0,  y0=0.0, v_y0 = 0.0, z0 = 100.0, v_z0 = 0.0, tf = 10.0, dt = 0.01, tag = "tag", omegax = 0.0, omegay = 0.0, omegaz = 0.0, masstocharge = 1):
        self.m = m
        self.q = 1 / masstocharge
        super().__init__()
    def mass_to_charge(self, m2q):
        self.q = 1 / m2q
    def force()
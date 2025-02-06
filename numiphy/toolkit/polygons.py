import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl


class Polygon:
    def __init__(self, x, y):

        self.n = len(x)
        self.coef = 4*self.n*np.tan(np.pi/self.n)

        self.x, self.y = np.array(x), np.array(y)
        self.r = [vector(x[i], y[i]) for i in range(self.n)]
        self.dr = [self.r[i]-self.r[i-1] for i in range(self.n)]
        self.area = 1/2*sum([self.r[i-1].cross(self.r[i]) for i in range(self.n)])
        self.per = sum([side.norm() for side in self.dr])
        self.R = self.coef*self.area/self.per**2

    def centroid(self):
        '''
        Geometric center:   x = \int x dxdy
                            y = \int y dxdy
        Using Green's Theorem, the above surface integral can be transformed to a line integral.
        '''
        x, y = self.x, self.y
        xc = 1/6*sum([(y[i] - y[i-1])*(x[i]**2 + x[i-1]**2 + x[i]*x[i-1]) for i in range(self.n)])/self.area
        yc = 1/6*sum([(x[i-1] - x[i])*(y[i]**2 + y[i-1]**2 + y[i]*y[i-1]) for i in range(self.n)])/self.area
        return np.array([xc, yc])

    def mean(self):
        '''
        average position = Σ r_i
        '''
        mean = self.r[0]
        for r in self.r[1:]:
            mean += r
        return mean.toarray()/self.n
    
    def grad_A(self, i: int):
        '''
        The surface derivative with respect to the i-th node
        '''
        return 0.5*(self.r[(i+1)%self.n]-self.r[i-1]).cross('z').toarray()
    
    def grad_P(self, i: int):
        '''
        The perimeter derivative with respect to the i-th node
        '''
        return (self.dr[i].unit()-self.dr[(i+1)%self.n].unit()).toarray()
    
    def grad_R(self, i: int):
        '''
        The roundness derivative with respect to the i-th node
        '''
        return self.coef*(self.grad_A(i)/self.per**2 - 2*self.area*self.grad_P(i)/self.per**3)
    
    def is_folded(self):
        if self.area < 0:
            return True
        elif self.n == 4:
            for i in range(self.n):
                A, B = self.r[i], self.r[(i+1)%self.n]
                for j in range(i+2, self.n-(i==0)):
                    C, D = self.r[j%self.n], self.r[(j+1)%self.n]
                    det = (C-B).cross(A-D)
                    if det == 0:
                        return False
                    t1 = (C-B).cross(B-D)/det
                    t2 = (A-D).cross(B-D)/det
                    if 0 < t1 < 1 and 0 < t2 < 1:
                        return True
        return False
    
    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(np.append(self.x, self.x[0]), np.append(self.y, self.y[0]))
        return fig, ax
    
    def contains(self, x, y):
        
        z = self.x-x + 1j*(self.y-y)
        if np.any(z==0):
            return True
        angle = 0
        for i in range(len(z)):
            angle += np.angle(z[i]/z[i-1])
        return abs(angle) > 1 #angle will be 2*pi


class vector:
    '''
    Use only for nodes in Polygon class
    '''
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __neg__(self):
        return vector(-self.x, -self.y)

    def __add__(self, other):
        return vector(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return vector(self.x-other.x, self.y-other.y)

    def __rmul__(self,other):
        return vector(other*self.x,other*self.y)

    def norm(self):
        return math.sqrt(self.x**2+self.y**2)

    def unit(self):
        norm = self.norm()
        return vector(self.x/norm, self.y/norm)

    def cross(self, other):
        if other == 'z':
            return vector(self.y, -self.x)
        else:
            return self.x*other.y - self.y*other.x

    def toarray(self):
        return np.array([self.x, self.y])

    def __repr__(self):
        r = [self.x, self.y]
        r = [int(i) if int(i)==i else i for i in r]
        return '({0}, {1})'.format(*r)


def circle_polygon(r, x, y, n):

    phi = np.linspace(0, 2*math.pi, n)
    x = x + r*np.cos(phi)
    y = y + r*np.sin(phi)
    return Polygon(x, y)

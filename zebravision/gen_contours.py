from numpy import cos,sin,arccos
import numpy as np

def parametric_circle(t,xc,yc,R):
    x = xc + R*cos(t)
    y = yc + R*sin(t)
    return x,y

def inv_parametric_circle(x,xc,R):
    t = arccos((x-xc)/R)
    return t

N = 30
R = 0.6
xc = .381/2
yc = -.52

start_point = (0, 0.050)
end_point   = (0.381, 0.050)

start_t = inv_parametric_circle(start_point[0], xc, R)
end_t   = inv_parametric_circle(end_point[0], xc, R)

arc_T = np.linspace(start_t, end_t, N)

from pylab import *
X,Y = parametric_circle(arc_T, xc, yc, R)

#X = X[::-1]
#Y = Y[::-1]

for x_p, y_p in zip(X,Y):
    print("contour_.push_back(Point2f(%s,%s));" %(x_p,y_p))

plot(X,Y)
scatter(X,Y)
scatter([xc],[yc],color='r',s=100)
axis('equal')
show()
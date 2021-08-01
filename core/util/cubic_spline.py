"""
Cubic spline planner
Author: Atsushi Sakai(@Atsushi_twi)
Source: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/CubicSpline/cubic_spline_planner.py
"""
import math
import numpy as np
import bisect


class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.eps = np.finfo(float).eps

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i] + self.eps))
            tb = (self.a[i + 1] - self.a[i]) / (h[i] + self.eps) - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / (h[i + 1] + self.eps) \
                       - 3.0 * (self.a[i + 1] - self.a[i]) / (h[i] + self.eps)
        return B


class Spline2D:
    """
    2D Cubic Spline class
    """

    def __init__(self, x, y, resolution=0.1):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

        self.s_fine = np.arange(0, self.s[-1], resolution)
        xy = np.array([self.calc_global_position_online(s_i) for s_i in self.s_fine])

        self.x_fine = xy[:, 0]
        self.y_fine = xy[:, 1]

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_global_position_online(self, s):
        """
        calc global position of points on the line, s: float
        return: x: float; y: float; the global coordinate of given s on the spline
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_global_position_offline(self, s, d):
        """
        calc global position of points in the frenet coordinate w.r.t. the line.
        s: float, longitudinal; d: float, lateral;
        return: x, float; y, float;
        """
        s_x = self.sx.calc(s)
        s_y = self.sy.calc(s)

        theta = math.atan2(self.sy.calcd(s), self.sx.calcd(s))
        x = s_x - math.sin(theta) * d
        y = s_y + math.cos(theta) * d
        return x, y

    def calc_frenet_position(self, x, y):
        """
        cal the frenet position of given global coordinate (x, y)
        return s: the longitudinal; d: the lateral
        """
        # find nearst x, y
        diff = np.hypot(self.x_fine - x, self.y_fine - y)
        idx = np.argmin(diff)
        [x_s, y_s] = self.x_fine[idx], self.y_fine[idx]
        s = self.s_fine[idx]

        # compute theta
        theta = math.atan2(self.sy.calcd(s), self.sx.calcd(s))
        d_x, d_y = x - x_s, y - y_s
        cross_rd_nd = math.cos(theta) * d_y - math.sin(theta) * d_x
        d = math.copysign(np.hypot(d_x, d_y), cross_rd_nd)
        return s, d

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw


def calc_spline_course(x, y, ds=0.1):
    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_global_position_online(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


def main():  # pragma: no cover
    print("Spline 2D test")
    import matplotlib.pyplot as plt
    # x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    # y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    x = [float(i) for i in range(30)]
    y = [float(0.0) for i in range(30)]
    for i in range(len(y)):
        if i % 2 == 0.0:
            y[i] = float(i)

    ds = 0.1  # [m] distance of each intepolated points

    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_global_position_online(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    plt.subplots(1)
    plt.plot(x, y, "xb", label="input")
    plt.plot(rx, ry, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.subplots(1)
    plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()


if __name__ == '__main__':
    main()

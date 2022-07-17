import numpy
import numpy as np
from scipy.integrate import odeint


def pdesolver(n: int, a: float, b: float, u: float,
              alpha1: float, alpha2: float, alpha3: float,
              beta1: float, beta2: float, beta3: float,
              c0: numpy.ndarray, tspan: numpy.ndarray) -> numpy.ndarray:
    """

    :param n: n+1 is the number of mesh points on the x-axis (i.e. x0, x1, ..., xn)
    :param a: the left boundary on the x-axis
    :param b: the right boundary on the x-axis
    :param u: the constant that describes the flow
    :param alpha1: parameters of the boundary condition on x = a
    :param alpha2: parameters of the boundary condition on x = a
    :param alpha3: parameters of the boundary condition on x = a
    :param beta1: parameters of the boundary condition on x = b
    :param beta2: parameters of the boundary condition on x = b
    :param beta3: parameters of the boundary condition on x = b
    :param c0: initial conditions across x in [a, b]
    :param tspan: the time interval
    :return: the numerical solution of the advection-diffusion equation
    """
    if u > 0:
        if alpha2 == 0:
            csol = odeint(posflowzero, c0, tspan, args=(n, a, b, u, alpha1, alpha2, alpha3, beta1, beta2, beta3))
            dx = (b - a) / n
            csol[:, n] = (beta2 * csol[:, n - 1] + beta3 * dx) / (beta2 + beta1 * dx)
        else:
            csol = odeint(posflow, c0, tspan, args=(n, a, b, u, alpha1, alpha2, alpha3, beta1, beta2, beta3))
    else:
        if beta2 == 0:
            csol = odeint(negflowzero, c0, tspan, args=(n, a, b, u, alpha1, alpha2, alpha3, beta1, beta2, beta3))
        else:
            csol = odeint(negflow, c0, tspan, args=(n, a, b, u, alpha1, alpha2, alpha3, beta1, beta2, beta3))
    return csol


def negflow(c: numpy.ndarray, t, n: int, a: float, b: float, u: float,
            alpha1: float, alpha2: float, alpha3: float,
            beta1: float, beta2: float, beta3: float) -> numpy.ndarray:
    """

    :param c: the variable of interest
    :param t: the time variable
    :param n: n+1 is the number of mesh points on the x-axis (i.e. x0, x1, ..., xn)
    :param a: the left boundary on the x-axis
    :param b: the right boundary on the x-axis
    :param u: the constant that describes the flow
    :param alpha1: parameters of the boundary condition on x = a
    :param alpha2: parameters of the boundary condition on x = a
    :param alpha3: parameters of the boundary condition on x = a
    :param beta1: parameters of the boundary condition on x = b
    :param beta2: parameters of the boundary condition on x = b
    :param beta3: parameters of the boundary condition on x = b
    :return: the transformed ODE system
    """
    dcdt = np.zeros(n + 1)
    dx = (b - a) / n
    c[0] = (alpha2 * c[1] - alpha3 * dx) / (alpha2 - alpha1 * dx)
    for i in range(n + 1):
        if i == 0:
            dcdt[i] = 0
        elif i == n:
            dcdt[i] = (((beta2 - beta1 * dx) / beta2 * c[i] + beta3 / beta2 * dx - 2 * c[i] + c[i - 1]) / (dx ** 2)
                       - u * ((beta3 - beta1 * c[i]) / beta2))
        else:
            dcdt[i] = (c[i + 1] - 2 * c[i] + c[i - 1]) / (dx ** 2) - u * (c[i + 1] - c[i]) / dx
    return dcdt


def posflow(c: numpy.ndarray, t, n: int, a: float, b: float, u: float,
            alpha1: float, alpha2: float, alpha3: float,
            beta1: float, beta2: float, beta3: float) -> numpy.ndarray:
    """

    :param c: the variable of interest
    :param t: the time variable
    :param n: n+1 is the number of mesh points on the x-axis (i.e. x0, x1, ..., xn)
    :param a: the left boundary on the x-axis
    :param b: the right boundary on the x-axis
    :param u: the constant that describes the flow
    :param alpha1: parameters of the boundary condition on x = a
    :param alpha2: parameters of the boundary condition on x = a
    :param alpha3: parameters of the boundary condition on x = a
    :param beta1: parameters of the boundary condition on x = b
    :param beta2: parameters of the boundary condition on x = b
    :param beta3: parameters of the boundary condition on x = b
    :return: the transformed ODE system
    """
    dcdt = np.zeros(n + 1)
    dx = (b - a) / n
    c[n] = (beta2 * c[n - 1] + beta3 * dx) / (beta2 + beta1 * dx)
    for i in range(n + 1):
        if i == n:
            dcdt[i] = 0
        elif i == 0:
            dcdt[i] = ((c[i + 1] - 2 * c[i] + (alpha2 + alpha1 * dx) / alpha2 * c[i] - alpha3 / alpha2 * dx) / (dx ** 2)
                       - u * ((alpha3 - alpha1 * c[i]) / alpha2))
        else:
            dcdt[i] = (c[i + 1] - 2 * c[i] + c[i - 1]) / (dx ** 2) - u * (c[i] - c[i - 1]) / dx
    return dcdt


def negflowzero(c: numpy.ndarray, t, n: int, a: float, b: float, u: float,
                alpha1: float, alpha2: float, alpha3: float,
                beta1: float, beta2: float, beta3: float) -> numpy.ndarray:
    """

    :param c: the variable of interest
    :param t: the time variable
    :param n: n+1 is the number of mesh points on the x-axis (i.e. x0, x1, ..., xn)
    :param a: the left boundary on the x-axis
    :param b: the right boundary on the x-axis
    :param u: the constant that describes the flow
    :param alpha1: parameters of the boundary condition on x = a
    :param alpha2: parameters of the boundary condition on x = a
    :param alpha3: parameters of the boundary condition on x = a
    :param beta1: parameters of the boundary condition on x = b
    :param beta2: parameters of the boundary condition on x = b, here equals 0
    :param beta3: parameters of the boundary condition on x = b
    :return: the transformed ODE system
    """
    dcdt = np.zeros(n + 1)
    dx = (b - a) / n
    c[0] = (alpha2 * c[1] - alpha3 * dx) / (alpha2 - alpha1 * dx)
    c[n] = beta3 / beta1
    for i in range(n + 1):
        if i == 0 or i == n:
            dcdt[i] = 0
        else:
            dcdt[i] = (c[i + 1] - 2 * c[i] + c[i - 1]) / (dx ** 2) - u * (c[i + 1] - c[i]) / dx
    return dcdt


def posflowzero(c: numpy.ndarray, t, n: int, a: float, b: float, u: float,
                alpha1: float, alpha2: float, alpha3: float,
                beta1: float, beta2: float, beta3: float) -> numpy.ndarray:
    """

    :param c: the variable of interest
    :param t: the time variable
    :param n: n+1 is the number of mesh points on the x-axis (i.e. x0, x1, ..., xn)
    :param a: the left boundary on the x-axis
    :param b: the right boundary on the x-axis
    :param u: the constant that describes the flow
    :param alpha1: parameters of the boundary condition on x = a
    :param alpha2: parameters of the boundary condition on x = a, here equals 0
    :param alpha3: parameters of the boundary condition on x = a
    :param beta1: parameters of the boundary condition on x = b
    :param beta2: parameters of the boundary condition on x = b
    :param beta3: parameters of the boundary condition on x = b
    :return: the transformed ODE system
    """
    dcdt = np.zeros(n + 1)
    dx = (b - a) / n
    c[0] = alpha3 / alpha1
    c[n] = (beta2 * c[n - 1] + beta3 * dx) / (beta2 + beta1 * dx)
    for i in range(n + 1):
        if i == 0 or i == n:
            dcdt[i] = 0
        else:
            dcdt[i] = (c[i + 1] - 2 * c[i] + c[i - 1]) / (dx ** 2) - u * (c[i] - c[i - 1]) / dx
    return dcdt

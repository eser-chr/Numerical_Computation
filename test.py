import numpy as np
import unittest

def globalised_descent_Newton(x0:np.ndarray, f:np.ndarray, df:np.ndarray, max_iter = 100, tol = np.finfo(float).eps, 
                              mu:float=0.5, q:float = 0.5):
    l = 1
    n = 0
    err = 1.0
    x = x0
    while(n<max_iter and err>tol):
        direction = np.linalg.solve(df(x), f(x))
        while(np.linalg.norm(f(x)) - np.linalg.norm(f(x+l*direction)) < mu * l *np.linalg.norm(f(x))):
            l = l*q
        x_new = x + l*direction
        l = min(1, l/q)
        err = np.linalg.norm(x_new-x)
        x=x_new
        n+=1
    return x


class TestGlobalisedDescentNewton(unittest.TestCase):

    def test_convergence(self):
        x0 = np.array([1.0, 1.0])
        f = lambda x: np.array([x[0]**2 + x[1]**2 - 1.0, x[0] - x[1]])
        df = lambda x: np.array([[2*x[0], 2*x[1]], [1, -1]])

        result = globalised_descent_Newton(x0, f, df)
        expected_result = np.array([1.0, 0.0])  # Replace with the expected result for your specific problem
        self.assertTrue(np.allclose(result, expected_result))


    def test_custom_tolerance(self):
        x0 = np.array([1.0, 1.0])
        f = lambda x: np.array([x[0]**2 + x[1]**2 - 1.0, x[0] - x[1]])
        df = lambda x: np.array([[2*x[0], 2*x[1]], [1, -1]])

        result = globalised_descent_Newton(x0, f, df, tol=1e-6)
        expected_result = np.array([1.0, 0.0])  # Replace with the expected result for your specific problem
        self.assertTrue(np.allclose(result, expected_result))

if __name__=="__main__":
    unittest.main()
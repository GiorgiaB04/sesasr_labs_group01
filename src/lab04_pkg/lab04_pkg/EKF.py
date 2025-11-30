import numpy as np
from numpy.linalg import inv
import math
from lab04_pkg.velocity4ekf import velocity_motion_model_wrapper, jacobian_Gt, jacobian_Vt     
from lab04_pkg.landmark_model import landmark_model_jacobian
class RobotEKF:
    def __init__(self):
        self.eval_gux = velocity_motion_model_wrapper   # 5-state motion model wrapper
        self.eval_Gt =  jacobian_Gt 
        self.eval_Vt = jacobian_Vt

        self.dim_x = 3
        self.dim_u = 2

        self.mu = np.zeros((self.dim_x))
        self.Sigma = np.eye(self.dim_x)
        self.Mt = np.eye(self.dim_u) # motion noise matrix (alpha1*vt^2+alpha2*wt^2 0; 0 alpha3*vt^2+alpha4*wt^2)

        self._I = np.eye(self.dim_x)

    def predict(self, u, sigma_u, g_extra_args=()):
        """
        Update the state prediction using the control input u and compute the relative uncertainty ellipse
        Parameters
        ----------

        u : np.array
            command for this step.

        sigma_u : np.array
            std dev for each component of the command signal

        extra_args : tuple
            any additional required parameter: dt

        Modified variables:
            self.mu: the state prediction
            self.Sigma: the covariance matrix of the state prediction
        """
        # Update the state prediction evaluating the motion model
        self.mu = self.eval_gux(self.mu, u, sigma_u, *g_extra_args)
 
        args = (*self.mu, *u)
        # Update the covariance matrix of the state prediction,
        # you need to evaluate the Jacobians Gt and Vt

        Gt = self.eval_Gt(self.mu,u, *g_extra_args)
        Vt = self.eval_Vt(self.mu,u, *g_extra_args)
        self.Sigma = Gt @ self.Sigma @ Gt.T + Vt @ self.Mt @ Vt.T
       

    def update(self, z, eval_hx, eval_Ht, Qt, Ht_args=(), hx_args=(),  residual=np.subtract, **kwargs):
        """Performs the update innovation of the extended Kalman filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.

        lmark : [x, y] list-like
            Landmark location in cartesian coordinates.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """

        # Convert the measurement to a vector if necessary. Needed for the residual computation
        if np.isscalar(z):
            z = np.asarray([z], float)
            Qt = np.atleast_2d(Qt).astype(float)
            Ht = np.atleast_2d(Ht).astype(float)

        # Compute the Kalman gain, you need to evaluate the Jacobian Ht
        Ht = eval_Ht(*Ht_args)
        SigmaHT = self.Sigma @ Ht.T
        self.S = Ht @ SigmaHT + Qt
        self.K = SigmaHT @ np.linalg.pinv(self.S)


        # Evaluate the expected measurement and compute the residual, then update the state prediction
        z_hat = eval_hx(*hx_args)
        if np.isscalar(z_hat):
            z_hat = np.asarray([z_hat], float)

        # if the z measurement include an angle update, we need to specify the positional index to normalize the residual
        y = residual(z, z_hat, **kwargs)
        self.mu = self.mu + self.K @ y

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        # Note that I is the identity matrix.
        I_KH = self._I - self.K @ Ht
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + self.K @ Qt @ self.K.T
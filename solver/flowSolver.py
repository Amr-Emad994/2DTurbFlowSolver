import numpy as np
import math


class CFD:
    def __init__(self):
        # --- Physical and numerical parameters (default values as in Fortran) ---
        self.rho = 1.0
        self.omegam = 0.5
        self.omegap = 0.3
        self.mudynamic = 1.0 / 40000.0
        self.sigk = 1.0
        self.omegak = 0.3
        self.omegaeps = 0.3
        self.cep1 = 1.44
        self.cep2 = 1.92
        self.cmu = 0.09
        self.sigeps = 1.3
        self.kappa = 0.41
        self.B = 5.5
        self.E = None  # Will be set in initialize as E = exp(kappa * B)
        self.ti = 0.1
        self.uinlet = 1.0
        self.icyclic = 0

        # Domain boundaries (set in initialize)
        self.xmin = 0.0
        self.xmax = 10.0
        self.ymin = 0.0
        self.ymax = 1.0
        self.dx = None
        self.dy = None

        # These will be set during initialization:
        self.nx = None
        self.ny = None
        self.maxiter = None

        # Declare arrays (set later in initialize)
        self.u = None;
        self.v = None;
        self.p = None;
        self.pp = None
        self.uold = None;
        self.vold = None
        self.apu = None;
        self.apv = None;
        self.app = None
        self.ae = None;
        self.aw = None;
        self.an = None;
        self.as_ = None  # use as_ since as is reserved
        self.source = None;
        self.dpdx = None;
        self.dpdy = None
        self.dpdxc = None;
        self.dpdyc = None;
        self.dpdxave = None;
        self.dpdyave = None
        self.ustar = None;
        self.vstar = None
        self.mut = None;
        self.sx = None;
        self.sy = None
        self.production = None;
        self.k = None;
        self.eps = None
        self.kold = None;
        self.epsold = None;
        self.dissipation = None
        self.apk = None;
        self.apeps = None;
        self.s = None

        # Grid arrays
        self.xp = None;
        self.xf = None;
        self.yp = None;
        self.yf = None
        self.fx = None;
        self.fy = None

    def initialize(self):
        # Set domain size (already defined in __init__)
        print("Enter number of cells in x-direction:")
        self.nx = int(input())
        print("Enter number of cells in y-direction:")
        self.ny = int(input())
        print("Enter number of outer iterations:")
        self.maxiter = int(input())

        nxx = self.nx + 2  # including boundary cells
        nyy = self.ny + 2

        # Allocate arrays (using np.zeros with appropriate shape)
        shape = (nxx, nyy)
        self.u = np.zeros(shape)
        self.v = np.zeros(shape)
        self.p = np.zeros(shape)
        self.pp = np.zeros(shape)
        self.uold = np.zeros(shape)
        self.vold = np.zeros(shape)
        self.apu = np.ones(shape)
        self.apv = np.ones(shape)
        self.app = np.ones(shape)
        self.ae = np.zeros(shape)
        self.aw = np.zeros(shape)
        self.an = np.zeros(shape)
        self.as_ = np.zeros(shape)
        self.source = np.zeros(shape)
        self.dpdx = np.zeros(shape)
        self.dpdy = np.zeros(shape)
        self.dpdxc = np.zeros(shape)
        self.dpdyc = np.zeros(shape)
        self.dpdxave = np.zeros(shape)
        self.dpdyave = np.zeros(shape)
        self.ustar = np.zeros(shape)
        self.vstar = np.zeros(shape)
        self.mut = np.zeros(shape)
        self.sx = np.zeros(shape)
        self.sy = np.zeros(shape)
        self.production = np.zeros(shape)
        self.k = np.zeros(shape)
        self.eps = np.zeros(shape)
        self.kold = np.zeros(shape)
        self.epsold = np.zeros(shape)
        self.dissipation = np.zeros(shape)
        self.apk = np.zeros(shape)
        self.apeps = np.zeros(shape)
        self.s = np.zeros(shape)

        # Allocate grid arrays (size nx+2 for x and ny+2 for y)
        self.xp = np.zeros(nxx)
        self.xf = np.zeros(nxx)
        self.yp = np.zeros(nyy)
        self.yf = np.zeros(nyy)
        self.fx = np.zeros(nxx)
        self.fy = np.zeros(nyy)

        # Set constant E from log-law constant
        self.E = math.exp(self.kappa * self.B)

        # Initialize flow field: set inlet condition for u on interior j indices 1:ny
        self.u[:, 1:self.ny + 1] = self.uinlet
        self.uold = self.u.copy()
        self.vold = self.v.copy()

        # Initialize turbulence quantities at interior cells (j=1,...,ny)
        for i in range(nxx):
            for j in range(1, self.ny + 1):
                self.k[i, j] = 2.0 / 3.0 * (self.uinlet * self.ti) ** 2
                self.eps[i, j] = (self.cmu ** 0.75) * (self.k[i, j] ** 1.5) / (0.07 * (self.ymax - self.ymin))
                self.mut[i, j] = self.rho * self.cmu * self.k[i, j] ** 2 / self.eps[i, j]
        self.mut[:, 0] = 0.0
        self.mut[:, self.ny + 1] = 0.0
        self.kold = self.k.copy()
        self.epsold = self.eps.copy()

        # Grid Setup:
        # Calculate dx and dy based on the number of cells (control volumes)
        self.dx = (self.xmax - self.xmin) / float(self.nx)
        self.dy = (self.ymax - self.ymin) / float(self.ny)

        # Define cell faces for x-direction:
        # xf[0] = xmin; xf[nx+1] = xmax; interior faces are computed with spacing dx
        self.xf[0] = self.xmin
        self.xf[self.nx + 1] = self.xmax
        for i in range(1, self.nx + 1):
            self.xf[i] = self.xmin + self.dx * (i - 1)

        # Define cell faces for y-direction:
        self.yf[0] = self.ymin
        self.yf[self.ny + 1] = self.ymax
        for j in range(1, self.ny + 1):
            self.yf[j] = self.ymin + self.dy * (j - 1)

        # Compute cell centers:
        for i in range(1, self.nx + 1):
            self.xp[i] = 0.5 * (self.xf[i] + self.xf[i + 1])
        self.xp[0] = self.xmin
        self.xp[self.nx + 1] = self.xmax

        for j in range(1, self.ny + 1):
            self.yp[j] = 0.5 * (self.yf[j] + self.yf[j + 1])
        self.yp[0] = self.ymin
        self.yp[self.ny + 1] = self.ymax

        # Compute stretching factors for interior cells
        for i in range(1, self.nx + 1):
            # Avoid division by zero; for i==0 this is not used.
            self.fx[i] = (self.xp[i] - self.xf[i]) / (
                self.xp[i] - self.xp[i - 1] if self.xp[i] - self.xp[i - 1] != 0 else 1.0)
        for j in range(1, self.ny + 1):
            self.fy[j] = (self.yp[j] - self.yf[j]) / (
                self.yp[j] - self.yp[j - 1] if self.yp[j] - self.yp[j - 1] != 0 else 1.0)

        print("Initialization complete.")

    def umom(self):
        # u-momentum update
        for j in range(1, self.ny + 1):
            for i in range(1, self.nx + 1):
                mdote = self.rho * (self.yf[j + 1] - self.yf[j]) * (
                            (1.0 - self.fx[i + 1]) * self.u[i + 1, j] + self.fx[i + 1] * self.u[i, j])
                mdotw = self.rho * (self.yf[j + 1] - self.yf[j]) * (
                            (1.0 - self.fx[i]) * self.u[i, j] + self.fx[i] * self.u[i - 1, j])
                mdotn = self.rho * (self.xf[i + 1] - self.xf[i]) * (
                            (1.0 - self.fy[j + 1]) * self.v[i, j + 1] + self.fy[j + 1] * self.v[i, j])
                mdots = self.rho * (self.xf[i + 1] - self.xf[i]) * (
                            (1.0 - self.fy[j]) * self.v[i, j] + self.fy[j] * self.v[i, j - 1])
                mue = self.mudynamic + (1.0 - self.fx[i + 1]) * self.mut[i + 1, j] + self.fx[i + 1] * self.mut[i, j]
                muw = self.mudynamic + (1.0 - self.fx[i]) * self.mut[i, j] + self.fx[i] * self.mut[i - 1, j]
                mun = self.mudynamic + (1.0 - self.fy[j + 1]) * self.mut[i, j + 1] + self.fy[j + 1] * self.mut[i, j]
                mus = self.mudynamic + (1.0 - self.fy[j]) * self.mut[i, j] + self.fy[j] * self.mut[i, j - 1]

                self.ae[i, j] = max(-mdote, 0.0) + mue * (self.yf[j + 1] - self.yf[j]) / (self.xp[i + 1] - self.xp[i])
                self.aw[i, j] = max(mdotw, 0.0) + muw * (self.yf[j + 1] - self.yf[j]) / (self.xp[i] - self.xp[i - 1])
                self.an[i, j] = max(-mdotn, 0.0) + mun * (self.xf[i + 1] - self.xf[i]) / (self.yp[j + 1] - self.yp[j])
                self.as_[i, j] = max(mdots, 0.0) + mus * (self.xf[i + 1] - self.xf[i]) / (self.yp[j] - self.yp[j - 1])

        # Correct east/west boundaries (viscous terms)
        for j in range(1, self.ny + 1):
            mue = self.mudynamic + self.mut[self.nx + 1, j]
            self.ae[self.nx, j] = max(-self.rho * (self.yf[j + 1] - self.yf[j]) * self.u[self.nx + 1, j], 0.0) + \
                                  mue * (self.yf[j + 1] - self.yf[j]) / (self.xp[self.nx + 1] - self.xp[self.nx])
            muw = self.mudynamic + self.mut[0, j]
            self.aw[1, j] = max(self.rho * (self.yf[j + 1] - self.yf[j]) * self.u[0, j], 0.0) + \
                            muw * (self.yf[j + 1] - self.yf[j]) / (self.xp[1] - self.xp[0])

        # Set wall boundary diffusive terms to zero
        self.as_[:, 1] = 0.0
        self.an[:, self.ny] = 0.0

        # Compute apu coefficient for interior cells
        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                self.apu[i, j] = (self.ae[i, j] + self.aw[i, j] + self.an[i, j] + self.as_[i, j]) / self.omegam

        # Wall shear stress as source term (using wall functions)
        for i in range(1, self.nx + 1):
            # Lower wall (j=1)
            ystar = self.rho * (self.cmu ** 0.25) * math.sqrt(self.k[i, 1]) * (self.yp[1] - self.yp[0]) / self.mudynamic
            tauwall = self.rho * self.kappa * (self.cmu ** 0.25) * math.sqrt(self.k[i, 1]) * self.u[i, 1] / math.log(
                self.E * ystar)
            self.s[i, 1] = self.rho * (self.cmu ** 0.25) * math.sqrt(self.k[i, 1]) * (self.xf[i + 1] - self.xf[i]) / (
                        math.log(self.E * ystar) / self.kappa)
            # Upper wall (j=ny)
            ystar = self.rho * (self.cmu ** 0.25) * math.sqrt(self.k[i, self.ny]) * (
                        self.yp[self.ny + 1] - self.yp[self.ny]) / self.mudynamic
            tauwall = self.rho * self.kappa * (self.cmu ** 0.25) * math.sqrt(self.k[i, self.ny]) * self.u[
                i, self.ny] / math.log(self.E * ystar)
            self.s[i, self.ny] = self.rho * (self.cmu ** 0.25) * math.sqrt(self.k[i, self.ny]) * (
                        self.xf[i + 1] - self.xf[i]) / (math.log(self.E * ystar) / self.kappa)

        # Compute source terms for variable viscosity
        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                mue = (1.0 - self.fx[i + 1]) * self.mut[i + 1, j] + self.fx[i + 1] * self.mut[i, j] + self.mudynamic
                muw = (1.0 - self.fx[i]) * self.mut[i, j] + self.fx[i] * self.mut[i - 1, j] + self.mudynamic
                term1 = mue * (self.u[i + 1, j] - self.u[i, j]) / (self.xp[i + 1] - self.xp[i])
                term2 = muw * (self.u[i, j] - self.u[i - 1, j]) / (self.xp[i] - self.xp[i - 1])
                term3 = (self.mudynamic + self.mut[i, j + 1]) * (self.v[i + 1, j + 1] - self.v[i - 1, j + 1]) / (
                            self.xp[i + 1] - self.xp[i - 1])
                term4 = (self.mudynamic + self.mut[i, j - 1]) * (self.v[i + 1, j - 1] - self.v[i - 1, j - 1]) / (
                            self.xp[i + 1] - self.xp[i - 1])
                term_x = (term1 - term2) / (self.xf[i + 1] - self.xf[i])
                term_y = (term3 - term4) / (self.yp[j + 1] - self.yp[j - 1])
                volume = (self.xf[i + 1] - self.xf[i]) * (self.yf[j + 1] - self.yf[j])
                self.sx[i, j] = (term_x + term_y) * volume

        # Outer iteration loop for u-momentum
        for outer in range(10):
            massflux = 0.0
            for j in range(1, self.ny + 1):
                for i in range(1, self.nx + 1):
                    self.u[i, j] = (1.0 - self.omegam) * self.uold[i, j] + (1.0 / self.apu[i, j]) * (
                            self.ae[i, j] * self.u[i + 1, j] +
                            self.aw[i, j] * self.u[i - 1, j] +
                            self.an[i, j] * self.u[i, j + 1] +
                            self.as_[i, j] * self.u[i, j - 1] +
                            (self.xf[i + 1] - self.xf[i]) * (self.yf[j + 1] - self.yf[j]) *
                            (self.p[i - 1, j] - self.p[i + 1, j]) / (self.xp[i + 1] - self.xp[i - 1]) +
                            self.sx[i, j] - self.s[i, j]
                    )
            massflux = 0.0
            for j in range(1, self.ny + 1):
                massflux += self.rho * self.dy * self.u[self.nx, j]
            self.u[self.nx + 1, 1:self.ny + 1] = (1.0 / self.rho) * self.u[self.nx, 1:self.ny + 1] / massflux

        if self.icyclic == 1:
            self.u[0, :] = self.u[self.nx + 1, :]

    def vmom(self):
        # v-momentum update
        for j in range(1, self.ny + 1):
            for i in range(1, self.nx + 1):
                mdote = self.rho * (self.yf[j + 1] - self.yf[j]) * (
                            (1.0 - self.fx[i + 1]) * self.uold[i + 1, j] + self.fx[i + 1] * self.uold[i, j])
                mdotw = self.rho * (self.yf[j + 1] - self.yf[j]) * (
                            (1.0 - self.fx[i]) * self.uold[i, j] + self.fx[i] * self.uold[i - 1, j])
                mdotn = self.rho * (self.xf[i + 1] - self.xf[i]) * (
                            (1.0 - self.fy[j + 1]) * self.v[i, j + 1] + self.fy[j + 1] * self.v[i, j])
                mdots = self.rho * (self.xf[i + 1] - self.xf[i]) * (
                            (1.0 - self.fy[j]) * self.v[i, j] + self.fy[j] * self.v[i, j - 1])
                mue = (1.0 - self.fx[i + 1]) * self.mut[i + 1, j] + self.fx[i + 1] * self.mut[i, j] + self.mudynamic
                muw = (1.0 - self.fx[i]) * self.mut[i, j] + self.fx[i] * self.mut[i - 1, j] + self.mudynamic
                mun = (1.0 - self.fy[j + 1]) * self.mut[i, j + 1] + self.fy[j + 1] * self.mut[i, j] + self.mudynamic
                mus = (1.0 - self.fy[j]) * self.mut[i, j] + self.fy[j] * self.mut[i, j - 1] + self.mudynamic

                self.ae[i, j] = max(-mdote, 0.0) + mue * (self.yf[j + 1] - self.yf[j]) / (self.xp[i + 1] - self.xp[i])
                self.aw[i, j] = max(mdotw, 0.0) + muw * (self.yf[j + 1] - self.yf[j]) / (self.xp[i] - self.xp[i - 1])
                self.an[i, j] = max(-mdotn, 0.0) + mun * (self.xf[i + 1] - self.xf[i]) / (self.yp[j + 1] - self.yp[j])
                self.as_[i, j] = max(mdots, 0.0) + mus * (self.xf[i + 1] - self.xf[i]) / (self.yp[j] - self.yp[j - 1])

        for j in range(1, self.ny + 1):
            mue = self.mut[self.nx + 1, j] + self.mudynamic
            self.ae[self.nx, j] = max(-self.rho * (self.yf[j + 1] - self.yf[j]) * self.uold[self.nx + 1, j], 0.0) + \
                                  mue * (self.yf[j + 1] - self.yf[j]) / (self.xp[self.nx + 1] - self.xp[self.nx])
            muw = self.mut[0, j] + self.mudynamic
            self.aw[1, j] = max(self.rho * (self.yf[j + 1] - self.yf[j]) * self.uold[0, j], 0.0) + \
                            muw * (self.yf[j + 1] - self.yf[j]) / (self.xp[1] - self.xp[0])

        self.an[:, self.ny] = 0.0
        self.as_[:, 1] = 0.0

        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                self.apv[i, j] = (self.ae[i, j] + self.aw[i, j] + self.an[i, j] + self.as_[i, j]) / self.omegam

        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                mun = ((1.0 - self.fy[j + 1]) * self.mut[i, j + 1] + self.fy[j + 1] * self.mut[i, j]) + self.mudynamic
                mus = ((1.0 - self.fy[j]) * self.mut[i, j] + self.fy[j] * self.mut[i, j - 1]) + self.mudynamic
                term1 = mun * (self.v[i, j + 1] - self.v[i, j]) / (self.yp[j + 1] - self.yp[j])
                term2 = mus * (self.v[i, j] - self.v[i, j - 1]) / (self.yp[j] - self.yp[j - 1])
                term3 = (self.mudynamic + self.mut[i + 1, j]) * (self.uold[i + 1, j + 1] - self.uold[i + 1, j - 1]) / (
                            self.yp[j + 1] - self.yp[j - 1])
                term4 = (self.mudynamic + self.mut[i - 1, j]) * (self.uold[i - 1, j + 1] - self.uold[i - 1, j - 1]) / (
                            self.yp[j + 1] - self.yp[j - 1])
                term_x = (term1 - term2) / (self.yf[j + 1] - self.yf[j])
                term_y = (term3 - term4) / (self.xp[i + 1] - self.xp[i - 1])
                volume = (self.xf[i + 1] - self.xf[i]) * (self.yf[j + 1] - self.yf[j])
                self.sy[i, j] = (term_x + term_y) * volume

        for outer in range(10):
            massflux = 0.0
            for j in range(1, self.ny + 1):
                for i in range(1, self.nx + 1):
                    self.v[i, j] = (1.0 - self.omegam) * self.vold[i, j] + (1.0 / self.apv[i, j]) * (
                            self.ae[i, j] * self.v[i + 1, j] +
                            self.aw[i, j] * self.v[i - 1, j] +
                            self.an[i, j] * self.v[i, j + 1] +
                            self.as_[i, j] * self.v[i, j - 1] +
                            (self.yf[j + 1] - self.yf[j]) * (self.xf[i + 1] - self.xf[i]) *
                            (self.p[i, j - 1] - self.p[i, j + 1]) / (self.yp[j + 1] - self.yp[j - 1]) +
                            self.sy[i, j]
                    )
            self.v[self.nx + 1, 1:self.ny + 1] = self.v[self.nx, 1:self.ny + 1]

        if self.icyclic == 1:
            self.v[0, :] = self.v[self.nx + 1, :]

    def pressure(self, outeriter):
        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                self.ae[i, j] = self.rho * (self.yf[j + 1] - self.yf[j]) ** 2 * (
                    ((1.0 - self.fx[i + 1]) / self.apu[i + 1, j] + self.fx[i + 1] / self.apu[i, j])
                )
                self.aw[i, j] = self.rho * (self.yf[j + 1] - self.yf[j]) ** 2 * (
                    ((1.0 - self.fx[i]) / self.apu[i, j] + self.fx[i] / self.apu[i - 1, j])
                )
                self.an[i, j] = self.rho * (self.xf[i + 1] - self.xf[i]) ** 2 * (
                    ((1.0 - self.fy[j + 1]) / self.apv[i, j + 1] + self.fy[j + 1] / self.apv[i, j])
                )
                self.as_[i, j] = self.rho * (self.xf[i + 1] - self.xf[i]) ** 2 * (
                    ((1.0 - self.fy[j]) / self.apv[i, j] + self.fy[j] / self.apv[i, j - 1])
                )

        self.ae[self.nx, 1:self.ny + 1] = 0.0
        self.aw[1, 1:self.ny + 1] = 0.0
        self.an[1:self.nx + 1, self.ny] = 0.0
        self.as_[1:self.nx + 1, 1] = 0.0

        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                self.app[i, j] = self.ae[i, j] + self.aw[i, j] + self.an[i, j] + self.as_[i, j]
        self.app[1, 1] = 1.e30

        for j in range(1, self.ny + 1):
            for i in range(2, self.nx + 1):
                self.dpdx[i, j] = (self.p[i, j] - self.p[i - 1, j]) / (self.xp[i] - self.xp[i - 1])
            self.dpdx[1, j] = (self.p[1, j] - self.p[0, j]) / (self.xp[1] - self.xp[0])
            self.dpdx[self.nx + 1, j] = (self.p[self.nx + 1, j] - self.p[self.nx, j]) / (
                        self.xp[self.nx + 1] - self.xp[self.nx])

        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                self.dpdxc[i, j] = (self.p[i + 1, j] - self.p[i - 1, j]) / (self.xp[i + 1] - self.xp[i - 1])
        for j in range(1, self.ny + 1):
            for i in range(2, self.nx + 1):
                self.dpdxave[i, j] = 0.5 * (self.dpdxc[i, j] + self.dpdxc[i - 1, j])

        for j in range(1, self.ny + 1):
            for i in range(2, self.nx + 1):
                correction = (self.xp[i] - self.xp[i - 1]) * (self.yf[j + 1] - self.yf[j]) * (
                    ((1.0 - self.fx[i]) / self.apu[i, j] + self.fx[i] / self.apu[i - 1, j])
                ) * (self.dpdx[i, j] - self.dpdxave[i, j])
                self.ustar[i, j] = ((1.0 - self.fx[i]) * self.u[i, j] + self.fx[i] * self.u[i - 1, j]) - correction
            self.ustar[1, j] = self.u[0, j]
            self.ustar[self.nx + 1, j] = self.u[self.nx + 1, j]

        for i in range(1, self.nx + 1):
            for j in range(2, self.ny + 1):
                self.dpdy[i, j] = (self.p[i, j] - self.p[i, j - 1]) / (self.yp[j] - self.yp[j - 1])
            self.dpdy[i, 1] = (self.p[i, 1] - self.p[i, 0]) / (self.yp[1] - self.yp[0])
            self.dpdy[i, self.ny + 1] = (self.p[i, self.ny + 1] - self.p[i, self.ny]) / (
                        self.yp[self.ny + 1] - self.yp[self.ny])

        for j in range(1, self.ny + 1):
            for i in range(1, self.nx + 1):
                self.dpdyc[i, j] = (self.p[i, j + 1] - self.p[i, j - 1]) / (self.yp[j + 1] - self.yp[j - 1])
        for i in range(1, self.nx + 1):
            for j in range(2, self.ny + 1):
                self.dpdyave[i, j] = 0.5 * (self.dpdyc[i, j] + self.dpdyc[i, j - 1])

        for i in range(1, self.nx + 1):
            for j in range(2, self.ny + 1):
                correction = (self.yp[j] - self.yp[j - 1]) * (self.xf[i + 1] - self.xf[i]) * (
                    ((1.0 - self.fy[j]) / self.apv[i, j] + self.fy[j] / self.apv[i, j - 1])
                ) * (self.dpdy[i, j] - self.dpdyave[i, j])
                self.vstar[i, j] = ((1.0 - self.fy[j]) * self.v[i, j] + self.fy[j] * self.v[i, j - 1]) - correction
            self.vstar[i, 1] = self.v[i, 0]
            self.vstar[i, self.ny + 1] = self.v[i, self.ny + 1]

        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                self.source[i, j] = (self.yf[j + 1] - self.yf[j]) * self.rho * (
                            self.ustar[i + 1, j] - self.ustar[i, j]) + \
                                    (self.xf[i + 1] - self.xf[i]) * self.rho * (self.vstar[i, j + 1] - self.vstar[i, j])
        total = math.sqrt(np.sum(self.source[1:self.nx + 1, 1:self.ny + 1] ** 2))
        print("Outer iteration:", outeriter, "Total mass imbalance =", total)

        self.pp.fill(0.0)
        for iter in range(100):
            for j in range(1, self.ny + 1):
                for i in range(1, self.nx + 1):
                    self.pp[i, j] = (1.0 / self.app[i, j]) * (
                            self.ae[i, j] * self.pp[i + 1, j] +
                            self.aw[i, j] * self.pp[i - 1, j] +
                            self.an[i, j] * self.pp[i, j + 1] +
                            self.as_[i, j] * self.pp[i, j - 1] -
                            self.source[i, j]
                    )

        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                self.p[i, j] += self.omegap * self.pp[i, j]
                self.u[i, j] -= (1.0 / self.apu[i, j]) * (self.xf[i + 1] - self.xf[i]) * (self.yf[j + 1] - self.yf[j]) * \
                                (self.pp[i + 1, j] - self.pp[i - 1, j]) / (self.xp[i + 1] - self.xp[i - 1])
                self.v[i, j] -= (1.0 / self.apv[i, j]) * (self.yf[j + 1] - self.yf[j]) * (self.xf[i + 1] - self.xf[i]) * \
                                (self.pp[i, j + 1] - self.pp[i, j - 1]) / (self.yp[j + 1] - self.yp[j - 1])

        self.p[0, 1:self.ny + 1] = 0.5 * (3.0 * self.p[1, 1:self.ny + 1] - self.p[2, 1:self.ny + 1])
        self.p[self.nx + 1, 1:self.ny + 1] = 0.5 * (
                    3.0 * self.p[self.nx, 1:self.ny + 1] - self.p[self.nx - 1, 1:self.ny + 1])
        self.p[1:self.nx + 1, 0] = 0.5 * (3.0 * self.p[1:self.nx + 1, 1] - self.p[1:self.nx + 1, 2])
        self.p[1:self.nx + 1, self.ny + 1] = 0.5 * (
                    3.0 * self.p[1:self.nx + 1, self.ny] - self.p[1:self.nx + 1, self.ny - 1])

        self.uold = self.u.copy()
        self.vold = self.v.copy()

    def kinetic_energy(self):
        for j in range(1, self.ny + 1):
            for i in range(1, self.nx + 1):
                mdote = self.rho * (self.yf[j + 1] - self.yf[j]) * (
                            (1.0 - self.fx[i + 1]) * self.u[i + 1, j] + self.fx[i + 1] * self.u[i, j])
                mdotw = self.rho * (self.yf[j + 1] - self.yf[j]) * (
                            (1.0 - self.fx[i]) * self.u[i, j] + self.fx[i] * self.u[i - 1, j])
                mdotn = self.rho * (self.xf[i + 1] - self.xf[i]) * (
                            (1.0 - self.fy[j + 1]) * self.v[i, j + 1] + self.fy[j + 1] * self.v[i, j])
                mdots = self.rho * (self.xf[i + 1] - self.xf[i]) * (
                            (1.0 - self.fy[j]) * self.v[i, j] + self.fy[j] * self.v[i, j - 1])
                mue = (1.0 - self.fx[i + 1]) * self.mut[i + 1, j] + self.fx[i + 1] * self.mut[i, j] + self.mudynamic
                muw = (1.0 - self.fx[i]) * self.mut[i, j] + self.fx[i] * self.mut[i - 1, j] + self.mudynamic
                mun = (1.0 - self.fy[j + 1]) * self.mut[i, j + 1] + self.fy[j + 1] * self.mut[i, j] + self.mudynamic
                mus = (1.0 - self.fy[j]) * self.mut[i, j] + self.fy[j] * self.mut[i, j - 1] + self.mudynamic
                self.ae[i, j] = max(-mdote, 0.0) + mue / self.sigk * (self.yf[j + 1] - self.yf[j]) / (
                            self.xp[i + 1] - self.xp[i])
                self.aw[i, j] = max(mdotw, 0.0) + muw / self.sigk * (self.yf[j + 1] - self.yf[j]) / (
                            self.xp[i] - self.xp[i - 1])
                self.an[i, j] = max(-mdotn, 0.0) + mun / self.sigk * (self.xf[i + 1] - self.xf[i]) / (
                            self.yp[j + 1] - self.yp[j])
                self.as_[i, j] = max(mdots, 0.0) + mus / self.sigk * (self.xf[i + 1] - self.xf[i]) / (
                            self.yp[j] - self.yp[j - 1])

        for j in range(1, self.ny + 1):
            mue = self.mut[self.nx + 1, j] + self.mudynamic
            self.ae[self.nx, j] = max(-self.rho * (self.yf[j + 1] - self.yf[j]) * self.uold[self.nx + 1, j], 0.0) + \
                                  mue * (self.yf[j + 1] - self.yf[j]) / (self.xp[self.nx + 1] - self.xp[self.nx])
            muw = self.mut[0, j] + self.mudynamic
            self.aw[1, j] = max(self.rho * (self.yf[j + 1] - self.yf[j]) * self.uold[0, j], 0.0) + \
                            muw * (self.yf[j + 1] - self.yf[j]) / (self.xp[1] - self.xp[0])

        self.an[:, self.ny] = 0.0
        self.as_[:, 1] = 0.0

        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                self.apk[i, j] = (self.ae[i, j] + self.aw[i, j] + self.an[i, j] + self.as_[i, j]) / self.omegak

        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                s11 = (self.u[i + 1, j] - self.u[i - 1, j]) / (self.xp[i + 1] - self.xp[i - 1])
                s22 = (self.v[i, j + 1] - self.v[i, j - 1]) / (self.yp[j + 1] - self.yp[j - 1])
                s12 = 0.5 * ((self.u[i, j + 1] - self.u[i, j - 1]) / (self.yp[j + 1] - self.yp[j - 1]) +
                             (self.v[i + 1, j] - self.v[i - 1, j]) / (self.xp[i + 1] - self.xp[i - 1]))
                production = 2.0 * self.mut[i, j] * (s11 ** 2 + s22 ** 2 + s12 ** 2 + s12 ** 2)
                volume = (self.xf[i + 1] - self.xf[i]) * (self.yf[j + 1] - self.yf[j])
                self.production[i, j] = production * volume
                self.dissipation[i, j] = volume * self.rho * self.eps[i, j]

        for i in range(1, self.nx + 1):
            ystar = self.rho * (self.cmu ** 0.25) * math.sqrt(self.k[i, 1]) * (self.yp[1] - self.yp[0]) / self.mudynamic
            tauwall = self.rho * self.kappa * (self.cmu ** 0.25) * math.sqrt(self.k[i, 1]) * self.u[i, 1] / math.log(
                self.E * ystar)
            self.production[i, 1] = tauwall * (self.cmu ** 0.25) * math.sqrt(self.k[i, 1]) / (
                        self.kappa * (self.yp[1] - self.yp[0]))
            self.production[i, 1] *= (self.xf[i + 1] - self.xf[i]) * (self.yf[2] - self.yf[1])
            ystar = self.rho * (self.cmu ** 0.25) * math.sqrt(self.k[i, self.ny]) * (
                        self.yp[self.ny + 1] - self.yp[self.ny]) / self.mudynamic
            tauwall = self.rho * self.kappa * (self.cmu ** 0.25) * math.sqrt(self.k[i, self.ny]) * self.u[
                i, self.ny] / math.log(self.E * ystar)
            self.production[i, self.ny] = tauwall * (self.cmu ** 0.25) * math.sqrt(self.k[i, self.ny]) / (
                        self.kappa * (self.yp[self.ny + 1] - self.yp[self.ny]))
            self.production[i, self.ny] *= (self.xf[i + 1] - self.xf[i]) * (self.yf[self.ny + 1] - self.yf[self.ny])

        for outer in range(10):
            for j in range(1, self.ny + 1):
                for i in range(1, self.nx + 1):
                    self.k[i, j] = (1.0 - self.omegak) * self.kold[i, j] + (1.0 / self.apk[i, j]) * (
                            self.ae[i, j] * self.k[i + 1, j] +
                            self.aw[i, j] * self.k[i - 1, j] +
                            self.an[i, j] * self.k[i, j + 1] +
                            self.as_[i, j] * self.k[i, j - 1] +
                            self.production[i, j] - self.dissipation[i, j]
                    )
            self.k[self.nx + 1, :] = self.k[self.nx, :]
        if self.icyclic == 1:
            self.k[0, :] = self.k[self.nx + 1, :]
        self.kold = self.k.copy()

    def dissipate(self):
        for j in range(2, self.ny):
            for i in range(1, self.nx + 1):
                mdote = self.rho * (self.yf[j + 1] - self.yf[j]) * (
                            (1.0 - self.fx[i + 1]) * self.u[i + 1, j] + self.fx[i + 1] * self.u[i, j])
                mdotw = self.rho * (self.yf[j + 1] - self.yf[j]) * (
                            (1.0 - self.fx[i]) * self.u[i, j] + self.fx[i] * self.u[i - 1, j])
                mdotn = self.rho * (self.xf[i + 1] - self.xf[i]) * (
                            (1.0 - self.fy[j + 1]) * self.v[i, j + 1] + self.fy[j + 1] * self.v[i, j])
                mdots = self.rho * (self.xf[i + 1] - self.xf[i]) * (
                            (1.0 - self.fy[j]) * self.v[i, j] + self.fy[j] * self.v[i, j - 1])
                mue = (1.0 - self.fx[i + 1]) * self.mut[i + 1, j] + self.fx[i + 1] * self.mut[i, j] + self.mudynamic
                muw = (1.0 - self.fx[i]) * self.mut[i, j] + self.fx[i] * self.mut[i - 1, j] + self.mudynamic
                mun = (1.0 - self.fy[j + 1]) * self.mut[i, j + 1] + self.fy[j + 1] * self.mut[i, j] + self.mudynamic
                mus = (1.0 - self.fy[j]) * self.mut[i, j] + self.fy[j] * self.mut[i, j - 1] + self.mudynamic
                self.ae[i, j] = max(-mdote, 0.0) + mue / self.sigeps * (self.yf[j + 1] - self.yf[j]) / (
                            self.xp[i + 1] - self.xp[i])
                self.aw[i, j] = max(mdotw, 0.0) + muw / self.sigeps * (self.yf[j + 1] - self.yf[j]) / (
                            self.xp[i] - self.xp[i - 1])
                self.an[i, j] = max(-mdotn, 0.0) + mun / self.sigeps * (self.xf[i + 1] - self.xf[i]) / (
                            self.yp[j + 1] - self.yp[j])
                self.as_[i, j] = max(mdots, 0.0) + mus / self.sigeps * (self.xf[i + 1] - self.xf[i]) / (
                            self.yp[j] - self.yp[j - 1])

        for j in range(2, self.ny):
            mue = self.mut[self.nx + 1, j] + self.mudynamic
            self.ae[self.nx, j] = max(-self.rho * (self.yf[j + 1] - self.yf[j]) * self.uold[self.nx + 1, j], 0.0) + \
                                  mue * (self.yf[j + 1] - self.yf[j]) / (self.xp[self.nx + 1] - self.xp[self.nx])
            muw = self.mut[0, j] + self.mudynamic
            self.aw[1, j] = max(self.rho * (self.yf[j + 1] - self.yf[j]) * self.uold[0, j], 0.0) + \
                            muw * (self.yf[j + 1] - self.yf[j]) / (self.xp[1] - self.xp[0])

        for i in range(1, self.nx + 1):
            for j in range(2, self.ny):
                self.apeps[i, j] = (self.ae[i, j] + self.aw[i, j] + self.an[i, j] + self.as_[i, j]) / self.omegaeps

        for i in range(1, self.nx + 1):
            for j in range(2, self.ny):
                s11 = (self.u[i + 1, j] - self.u[i - 1, j]) / (self.xp[i + 1] - self.xp[i - 1])
                s22 = (self.v[i, j + 1] - self.v[i, j - 1]) / (self.yp[j + 1] - self.yp[j - 1])
                s12 = 0.5 * ((self.u[i, j + 1] - self.u[i, j - 1]) / (self.yp[j + 1] - self.yp[j - 1]) +
                             (self.v[i + 1, j] - self.v[i - 1, j]) / (self.xp[i + 1] - self.xp[i - 1]))
                prod = self.cep1 * self.eps[i, j] / self.k[i, j] * 2.0 * self.mut[i, j] * (
                            s11 ** 2 + s22 ** 2 + s12 ** 2 + s12 ** 2)
                volume = (self.xf[i + 1] - self.xf[i]) * (self.yf[j + 1] - self.yf[j])
                self.production[i, j] = prod * volume
                self.dissipation[i, j] = volume * self.cep2 * self.rho * (self.eps[i, j] ** 2) / self.k[i, j]

        for outer in range(10):
            for j in range(2, self.ny):
                for i in range(1, self.nx + 1):
                    self.eps[i, j] = (1.0 - self.omegaeps) * self.epsold[i, j] + (1.0 / self.apeps[i, j]) * (
                            self.ae[i, j] * self.eps[i + 1, j] +
                            self.aw[i, j] * self.eps[i - 1, j] +
                            self.an[i, j] * self.eps[i, j + 1] +
                            self.as_[i, j] * self.eps[i, j - 1] +
                            self.production[i, j] - self.dissipation[i, j]
                    )
            self.eps[self.nx + 1, :] = self.eps[self.nx, :]
            for i in range(1, self.nx + 1):
                self.eps[i, 1] = (self.cmu ** 0.75) * self.k[i, 1] ** 1.5 / (self.kappa * (self.yp[1] - self.yp[0]))
                self.eps[i, self.ny] = (self.cmu ** 0.75) * self.k[i, self.ny] ** 1.5 / (
                            self.kappa * (self.yp[self.ny + 1] - self.yp[self.ny]))
        if self.icyclic == 1:
            self.eps[0, :] = self.eps[self.nx + 1, :]
        self.epsold = self.eps.copy()

    def turb_visc(self):
        for i in range(1, self.nx + 1):
            for j in range(1, self.ny + 1):
                self.mut[i, j] = self.rho * self.cmu * self.k[i, j] ** 2 / self.eps[i, j]
        self.mut[self.nx + 1, :] = self.mut[self.nx, :]
        if self.icyclic == 1:
            self.mut[0, :] = self.mut[self.nx + 1, :]

    def output(self):
        with open('RESULTS1.csv', 'w') as f1, open('RESULTS2.csv', 'w') as f2:
            f1.write("x, y, z, pressure, velmag, vx, vy, vz\n")
            f2.write("x, y, z, k, eps, mut\n")
            self.u[0, 0] = 0;
            self.v[0, 0] = 0;
            self.p[0, 0] = self.p[1, 1]
            self.u[self.nx + 1, 0] = 0;
            self.v[self.nx + 1, 0] = 0;
            self.p[self.nx + 1, 0] = self.p[self.nx, 1]
            self.u[self.nx + 1, self.ny + 1] = 0;
            self.v[self.nx + 1, self.ny + 1] = 0;
            self.p[self.nx + 1, self.ny + 1] = self.p[self.nx, self.ny]
            self.u[0, self.ny + 1] = 0;
            self.v[0, self.ny + 1] = 0;
            self.p[0, self.ny + 1] = self.p[1, self.ny]
            for i in range(0, self.nx + 2):
                for j in range(0, self.ny + 2):
                    velmag = math.sqrt(self.u[i, j] ** 2 + self.v[i, j] ** 2)
                    f1.write(
                        f"{self.xp[i]:.5f}, {self.yp[j]:.5f}, 0.0, {self.p[i, j]:.5f}, {velmag:.5f}, {self.u[i, j]:.5f}, {self.v[i, j]:.5f}, 0.0\n")
                    f2.write(
                        f"{self.xp[i]:.5f}, {self.yp[j]:.5f}, 0.0, {self.k[i, j]:.5f}, {self.eps[i, j]:.5f}, {self.mut[i, j]:.5f}\n")
        umax = np.max(self.u[self.nx, :])
        print("Maximum centerline velocity =", umax)
        wall_ystar = self.rho * (self.cmu ** 0.25) * math.sqrt(self.k[self.nx, 1]) * (
                    self.yp[1] - self.yp[0]) / self.mudynamic
        print("Wall ystar =", wall_ystar)
        with open('wall_profile.txt', 'w') as fw:
            fw.write("y/ymax, u/u_max\n")
            for j in range(0, self.ny + 2):
                fw.write(f"{self.yp[j] / self.ymax:.5f}, {self.u[self.nx, j] / umax:.5f}\n")
        print("Output complete.")

    def run_simulation(self):
        self.initialize()
        print("Finished initialize.")
        for outer in range(1, self.maxiter + 1):
            self.umom()
            self.vmom()
            self.pressure(outer)
            self.kinetic_energy()
            self.dissipate()
            self.turb_visc()
        self.output()


if __name__ == '__main__':
    sim = CFD()
    sim.run_simulation()

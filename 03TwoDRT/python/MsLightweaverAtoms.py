from lightweaver.rh_atoms import He_9_atom
from lightweaver.atomic_model import reconfigure_atom, AtomicModel,  AtomicLine, AtomicLevel, VoigtLine, LineType, HydrogenicContinuum, ExplicitContinuum, LinearCoreExpWings
from lightweaver.broadening import VdwApprox, VdwUnsold, HydrogenLinearStarkBroadening, LineBroadening, RadiativeBroadening, MultiplicativeStarkBroadening
from lightweaver.collisional_rates import CollisionalRates, Omega, CI, CE, Ar85Cdi, Burgess, fone, TemperatureInterpolationRates
from lightweaver.atomic_set import SpeciesStateTable
import lightweaver as lw
from fractions import Fraction
from Fang import FangHRates
import lightweaver.constants as Const
import pickle
from dataclasses import dataclass
import numpy as np
from weno4 import weno4
        # tg = atmos.temperature

@dataclass(eq=False, repr=False)
class VdwRadyn(VdwApprox):
    def setup(self, line: AtomicLine):
        self.line = line
        if len(self.vals) != 1:
            raise ValueError('VdwRadyn expects 1 coefficient (%s)' % repr(line))

        Z = line.jLevel.stage + 1
        j = line.j
        ic = j + 1
        while line.atom.levels[ic].stage < Z:
            ic += 1
        cont = line.atom.levels[ic]

        zz = line.iLevel.stage + 1
        deltaR = (Const.ERydberg / (cont.E_SI - line.jLevel.E_SI))**2 \
                - (Const.ERydberg / (cont.E_SI - line.iLevel.E_SI))**2
        fourPiEps0 = 4.0 * np.pi * Const.Epsilon0
        c625 = (2.5 * Const.QElectron**2 / fourPiEps0 * Const.ABarH / fourPiEps0 \
                * 2 * np.pi * (Z * Const.RBohr)**2 / Const.HPlanck * deltaR)**0.4

        self.cross = self.vals[0] * 8.411 * (8.0 * Const.KBoltzmann / np.pi * \
            (1.0 / (lw.PeriodicTable['H'].mass * Const.Amu) + \
                1.0 / (line.atom.element.mass * Const.Amu)))**0.3 * c625


    def broaden(self, atmos: lw.Atmosphere, eqPops: SpeciesStateTable) -> np.ndarray:
        nHGround = eqPops['H'][0, :]
        return self.cross * atmos.temperature**0.3 * nHGround

# NOTE(cmo): Ignoring the continua (bf) grids for now, they shouldn't be important here, and would cause us to need to mess around with our cross-section grids

def convert_alphaGrid(alphaGrid):
    a = np.array(alphaGrid)
    result = {'wavelengthGrid': a[:, 0][::-1].tolist(),
              'alphaGrid': a[:, 1][::-1].tolist()}
    return result


def H_6():
    radynQNorm = 12.85e3
    qNormRatio = radynQNorm / lw.VMICRO_CHAR
    qr = qNormRatio

    H_6_radyn = lambda: \
    AtomicModel(element=lw.PeriodicTable['H'],
    levels=[
        AtomicLevel(E=0.000000, g=2.000000, label="H I 1S 2SE", stage=0, J=Fraction(1, 2), L=0, S=Fraction(1, 2)),
        AtomicLevel(E=82257.172000, g=8.000000, label="H I 2P 2PO", stage=0, J=Fraction(7, 2), L=1, S=Fraction(1, 2)),
        AtomicLevel(E=97489.992000, g=18.000000, label="H I 3D 2DE", stage=0, J=Fraction(17, 2), L=2, S=Fraction(1, 2)),
        AtomicLevel(E=102821.219000, g=32.000000, label="H I 4F 2FO", stage=0, J=Fraction(31, 2), L=3, S=Fraction(1, 2)),
        AtomicLevel(E=105288.859000, g=50.000000, label="H I 5G 2GE", stage=0, J=Fraction(49, 2), L=4, S=Fraction(1, 2)),
        AtomicLevel(E=109754.578000, g=1.000000, label="H II continuum", stage=1, J=None, L=None, S=None),
    ],
    lines=[
        VoigtLine(j=1, i=0, f=4.167000e-01, type=LineType.PRD, quadrature=LinearCoreExpWings(Nlambda=100, qCore=20.000000, qWing=600.000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=0.0)], elastic=[VdwRadyn(vals=[0.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=2, i=0, f=7.919000e-02, type=LineType.PRD, quadrature=LinearCoreExpWings(Nlambda=100, qCore=20.000000, qWing=250.000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=0.0)], elastic=[VdwRadyn(vals=[0.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=3, i=0, f=2.901000e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=10.000000, qWing=100.0000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=0.0)], elastic=[VdwRadyn(vals=[0.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=0, f=1.395000e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=10.000000, qWing=100.0000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=0.0)], elastic=[VdwRadyn(vals=[0.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=2, i=1, f=6.414000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=100, qCore=20.000000, qWing=250.000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=5.701e+8)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=3, i=1, f=1.195000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=80, qCore=10.000000, qWing=250.0000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=5.004e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=1, f=4.471000e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=80, qCore=10.000000, qWing=250.0000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.818e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=3, i=2, f=8.431000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=10.000000, qWing=30.00000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.301e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=2, f=1.508000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=10.000000, qWing=30.00000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.115e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=3, f=1.039000e+00, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=5.000000, qWing=30.000000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.177e+07)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        # NOTE(cmo): I don't believe RADYN does both linear and quadratic Stark broadening for H, so setting Stark to these numbers seems to make the lines overly wide
        # VoigtLine(j=1, i=0, f=4.167000e-01, type=LineType.PRD, NlambdaGen=100, qCore=15.000000, qWing=600.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=2, i=0, f=7.919000e-02, type=LineType.PRD, NlambdaGen=50, qCore=10.000000, qWing=250.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=3, i=0, f=2.901000e-02, type=LineType.CRD, NlambdaGen=20, qCore=3.000000, qWing=100.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=4, i=0, f=1.395000e-02, type=LineType.CRD, NlambdaGen=20, qCore=3.000000, qWing=100.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=2, i=1, f=6.414000e-01, type=LineType.CRD, NlambdaGen=70, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=5.701e+8, stark=1.026e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=3, i=1, f=1.195000e-01, type=LineType.CRD, NlambdaGen=40, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=5.004e+08, stark=3.836e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=4, i=1, f=4.471000e-02, type=LineType.CRD, NlambdaGen=40, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=4.818e+08, stark=6.715e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=3, i=2, f=8.431000e-01, type=LineType.CRD, NlambdaGen=20, qCore=2.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=1.301e+07, stark=3.212e-4*-Const.CM_TO_M**3),
        # VoigtLine(j=4, i=2, f=1.508000e-01, type=LineType.CRD, NlambdaGen=20, qCore=2.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=1.115e+08, stark=0.000000),
        # VoigtLine(j=4, i=3, f=1.039000e+00, type=LineType.CRD, NlambdaGen=20, qCore=1.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=4.177e+07, stark=0.000000),
    ],
    continua=[
        ExplicitContinuum(j=5, i=0, **convert_alphaGrid(alphaGrid=[[91.17, 6.538849999999999e-24],
                                                [85.0, 5.299099999999999e-24],
                                                [80.0, 4.41789e-24],
                                                [75.0, 3.64023e-24],
                                                [70.0, 2.95964e-24],
                                                [60.0, 1.8638e-24],
                                                [50.5, 1.1112699999999999e-24]])),
        ExplicitContinuum(j=5, i=1, **convert_alphaGrid(alphaGrid=[[364.69, 1.41e-23],
                                                [355.0, 1.3005699999999999e-23],
                                                [340.0, 1.14257e-23],
                                                [320.0, 9.52571e-24],
                                                [308.0, 8.493759999999999e-24],
                                                [299.2, 7.78632e-24],
                                                [283.0, 6.588809999999999e-24],
                                                [282.6, 6.56091e-24],
                                                [281.4, 6.47768e-24],
                                                [253.9, 4.75812e-24],
                                                [251.4, 4.61895e-24],
                                                [251.3, 4.61344e-24],
                                                [235.0, 3.77269e-24],
                                                [207.1, 2.5821899999999997e-24],
                                                [206.9, 2.5747099999999997e-24],
                                                [180.0, 1.6953699999999998e-24],
                                                [168.2, 1.3833299999999998e-24],
                                                [168.0, 1.3783999999999998e-24],
                                                [152.1, 1.0229e-24],
                                                [151.9, 1.01887e-24],
                                                [140.7, 8.097e-25],
                                                [138.9, 7.789999999999999e-25],
                                                [135.8, 7.28e-25],
                                                [133.2, 6.87e-25],
                                                [130.0, 6.3867199999999995e-25],
                                                [120.1, 5.035890000000001e-25],
                                                [119.9, 5.01078e-25],
                                                [110.1, 3.8798e-25],
                                                [109.9, 3.8586999999999997e-25],
                                                [91.2, 2.20512e-25]])),
        ExplicitContinuum(j=5, i=2, **convert_alphaGrid(alphaGrid=[[820.57, 2.18e-23],
                                                [780.0, 1.8723799999999998e-23],
                                                [740.0, 1.5988399999999998e-23],
                                                [700.0, 1.3533299999999999e-23],
                                                [669.0, 1.1813799999999999e-23],
                                                [668.4, 1.1781999999999999e-23],
                                                [579.0, 7.65853e-24],
                                                [555.05, 6.74692e-24],
                                                [513.0, 5.32674e-24],
                                                [470.0, 4.09641e-24],
                                                [450.45, 3.6062e-24],
                                                [430.0, 3.1369999999999998e-24],
                                                [417.0, 2.8609999999999998e-24],
                                                [400.0, 2.52516e-24],
                                                [387.0, 2.28687e-24],
                                                [364.71, 1.9140499999999999e-24]])),
        ExplicitContinuum(j=5, i=3, **convert_alphaGrid(alphaGrid=[[1442.2, 2.91e-23],
                                                [1350.0, 2.38681e-23],
                                                [1215.0, 1.7399799999999997e-23],
                                                [1000.0, 9.700999999999998e-24],
                                                [876.0, 6.52122e-24],
                                                [820.6, 5.36057e-24]])),
        ExplicitContinuum(j=5, i=4, **convert_alphaGrid(alphaGrid=[[2239.2, 3.63e-23],
                                                [2090.0, 2.95166e-23],
                                                [1730.0, 1.67404e-23],
                                                [1670.0, 1.5058299999999998e-23],
                                                [1540.0, 1.1808399999999999e-23],
                                                [1442.3, 9.700509999999999e-24]]))
    ],
    collisions=[
        Omega(j=1, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.61, 0.613, 0.63, 0.686, 0.847, 1.24, 2.02, 3.58, 163.0, 163.0, 163.0, 163.0, 163.0]),
        Omega(j=2, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.153, 0.155, 0.162, 0.179, 0.217, 0.296, 0.444, 0.74, 31.0, 31.0, 31.0, 31.0, 31.0]),
        Omega(j=3, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.0609, 0.0618, 0.0647, 0.0713, 0.0858, 0.115, 0.169, 0.277, 11.3, 11.3, 11.3, 11.3, 11.3]),
        Omega(j=4, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.0303, 0.0308, 0.0323, 0.0355, 0.0426, 0.0569, 0.0828, 0.135, 5.43, 5.43, 5.43, 5.43, 5.43]),
        Omega(j=2, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[25.8, 28.0, 36.4, 55.0, 89.5, 143.0, 214.0, 356.0, 14900.0, 14900.0, 14900.0, 14900.0, 14900.0]),
        Omega(j=3, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[8.04, 8.69, 10.7, 14.4, 20.8, 30.0, 41.9, 65.7, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0]),
        Omega(j=4, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[3.52, 3.78, 4.54, 5.96, 8.32, 11.7, 16.1, 24.9, 924.0, 924.0, 924.0, 924.0, 924.0]),
        Omega(j=3, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[275.0, 347.0, 561.0, 923.0, 1450.0, 2120.0, 2900.0, 4460.0, 164000.0, 164000.0, 164000.0, 164000.0, 164000.0]),
        Omega(j=4, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[88.4, 104.0, 145.0, 209.0, 295.0, 398.0, 515.0, 749.0, 24700.0, 24700.0, 24700.0, 24700.0, 24700.0]),
        Omega(j=4, i=3, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[1830.0, 2500.0, 4210.0, 6680.0, 9830.0, 13500.0, 17400.0, 25200.0, 822000.0, 822000.0, 822000.0, 822000.0, 822000.0]),
        CI(j=5, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[3.06e-17, 3.3199999999999996e-17, 3.82e-17, 4.3099999999999994e-17, 4.8e-17, 5.3e-17, 5.79e-17, 6.29e-17, 1.0099999999999999e-16, 8.91e-17, 6.75e-17, 4.61e-17, 2.94e-17, 1.7799999999999998e-17, 1.0499999999999998e-17, 1.0499999999999998e-17, 1.0499999999999998e-17]),
        CI(j=5, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[1.06e-15, 9e-16, 8.99e-16, 9.46e-16, 9.56e-16, 9.07e-16, 7.68e-16, 5.68e-16, 9.819999999999998e-16, 7.36e-16, 4.69e-16, 2.84e-16, 1.6699999999999998e-16, 9.57e-17, 5.39e-17, 5.39e-17, 5.39e-17]),
        CI(j=5, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[3.8e-15, 4.32e-15, 4.73e-15, 4.68e-15, 4.25e-15, 3.5199999999999996e-15, 2.55e-15, 1.5199999999999998e-15, 3.0699999999999998e-15, 2.18e-15, 1.31e-15, 7.67e-16, 4.3799999999999997e-16, 2.4599999999999995e-16, 1.36e-16, 1.36e-16, 1.36e-16]),
        CI(j=5, i=3, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[1.44e-14, 1.5e-14, 1.48e-14, 1.3399999999999998e-14, 1.13e-14, 8.49e-15, 5.23e-15, 1.66e-15, 6.53e-15, 4.5e-15, 2.64e-15, 1.5199999999999998e-15, 8.53e-16, 4.74e-16, 2.6099999999999995e-16, 2.6099999999999995e-16, 2.6099999999999995e-16]),
        CI(j=5, i=4, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[4.06e-14, 3.86e-14, 3.42e-14, 2.87e-14, 2.25e-14, 1.58e-14, 8.57e-15, 2.3e-15, 1.1399999999999999e-14, 7.77e-15, 4.49e-15, 2.54e-15, 1.42e-15, 7.829999999999999e-16, 4.29e-16, 4.29e-16, 4.29e-16]),
        FangHRates(0,0)
    ])

    H = H_6_radyn()
    for c in H.continua:
        for i, a in enumerate(c.alphaGrid):
            c.alphaGrid[i] *= 1e2
    # for l in H.lines:
    #     l.NlambdaGen *= 2
    # H.collisions.append(FangHRates(0,0))

    reconfigure_atom(H)
    # for r in radTrans:
        # if r['atomName'] == H.name:
            # if r['transType'] == 'Line':
                # H.lines[r['lwIdx']].wavelength = r['wlGrid'].value
                # H.lines[r['lwIdx']].preserveWavelength = True

    return H

def H_6_noLybb():
    radynQNorm = 12.85e3
    qNormRatio = radynQNorm / lw.VMICRO_CHAR
    qr = qNormRatio

    H_6_radyn = lambda: \
    AtomicModel(element=lw.PeriodicTable['H'],
    levels=[
        AtomicLevel(E=0.000000, g=2.000000, label="H I 1S 2SE", stage=0, J=Fraction(1, 2), L=0, S=Fraction(1, 2)),
        AtomicLevel(E=82257.172000, g=8.000000, label="H I 2P 2PO", stage=0, J=Fraction(7, 2), L=1, S=Fraction(1, 2)),
        AtomicLevel(E=97489.992000, g=18.000000, label="H I 3D 2DE", stage=0, J=Fraction(17, 2), L=2, S=Fraction(1, 2)),
        AtomicLevel(E=102821.219000, g=32.000000, label="H I 4F 2FO", stage=0, J=Fraction(31, 2), L=3, S=Fraction(1, 2)),
        AtomicLevel(E=105288.859000, g=50.000000, label="H I 5G 2GE", stage=0, J=Fraction(49, 2), L=4, S=Fraction(1, 2)),
        AtomicLevel(E=109754.578000, g=1.000000, label="H II continuum", stage=1, J=None, L=None, S=None),
    ],
    lines=[
        VoigtLine(j=2, i=1, f=6.414000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=100, qCore=20.000000, qWing=250.000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=5.701e+8)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=3, i=1, f=1.195000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=80, qCore=10.000000, qWing=250.0000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=5.004e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=1, f=4.471000e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=80, qCore=10.000000, qWing=250.0000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.818e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=3, i=2, f=8.431000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=10.000000, qWing=30.00000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.301e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=2, f=1.508000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=10.000000, qWing=30.00000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.115e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=3, f=1.039000e+00, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=5.000000, qWing=30.000000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.177e+07)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        # NOTE(cmo): I don't believe RADYN does both linear and quadratic Stark broadening for H, so setting Stark to these numbers seems to make the lines overly wide
        # VoigtLine(j=1, i=0, f=4.167000e-01, type=LineType.PRD, NlambdaGen=100, qCore=15.000000, qWing=600.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=2, i=0, f=7.919000e-02, type=LineType.PRD, NlambdaGen=50, qCore=10.000000, qWing=250.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=3, i=0, f=2.901000e-02, type=LineType.CRD, NlambdaGen=20, qCore=3.000000, qWing=100.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=4, i=0, f=1.395000e-02, type=LineType.CRD, NlambdaGen=20, qCore=3.000000, qWing=100.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=2, i=1, f=6.414000e-01, type=LineType.CRD, NlambdaGen=70, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=5.701e+8, stark=1.026e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=3, i=1, f=1.195000e-01, type=LineType.CRD, NlambdaGen=40, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=5.004e+08, stark=3.836e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=4, i=1, f=4.471000e-02, type=LineType.CRD, NlambdaGen=40, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=4.818e+08, stark=6.715e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=3, i=2, f=8.431000e-01, type=LineType.CRD, NlambdaGen=20, qCore=2.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=1.301e+07, stark=3.212e-4*-Const.CM_TO_M**3),
        # VoigtLine(j=4, i=2, f=1.508000e-01, type=LineType.CRD, NlambdaGen=20, qCore=2.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=1.115e+08, stark=0.000000),
        # VoigtLine(j=4, i=3, f=1.039000e+00, type=LineType.CRD, NlambdaGen=20, qCore=1.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=4.177e+07, stark=0.000000),
    ],
    continua=[
        ExplicitContinuum(j=5, i=0, **convert_alphaGrid(alphaGrid=[[91.17, 6.538849999999999e-24],
                                                [85.0, 5.299099999999999e-24],
                                                [80.0, 4.41789e-24],
                                                [75.0, 3.64023e-24],
                                                [70.0, 2.95964e-24],
                                                [60.0, 1.8638e-24],
                                                [50.5, 1.1112699999999999e-24]])),
        ExplicitContinuum(j=5, i=1, **convert_alphaGrid(alphaGrid=[[364.69, 1.41e-23],
                                                [355.0, 1.3005699999999999e-23],
                                                [340.0, 1.14257e-23],
                                                [320.0, 9.52571e-24],
                                                [308.0, 8.493759999999999e-24],
                                                [299.2, 7.78632e-24],
                                                [283.0, 6.588809999999999e-24],
                                                [282.6, 6.56091e-24],
                                                [281.4, 6.47768e-24],
                                                [253.9, 4.75812e-24],
                                                [251.4, 4.61895e-24],
                                                [251.3, 4.61344e-24],
                                                [235.0, 3.77269e-24],
                                                [207.1, 2.5821899999999997e-24],
                                                [206.9, 2.5747099999999997e-24],
                                                [180.0, 1.6953699999999998e-24],
                                                [168.2, 1.3833299999999998e-24],
                                                [168.0, 1.3783999999999998e-24],
                                                [152.1, 1.0229e-24],
                                                [151.9, 1.01887e-24],
                                                [140.7, 8.097e-25],
                                                [138.9, 7.789999999999999e-25],
                                                [135.8, 7.28e-25],
                                                [133.2, 6.87e-25],
                                                [130.0, 6.3867199999999995e-25],
                                                [120.1, 5.035890000000001e-25],
                                                [119.9, 5.01078e-25],
                                                [110.1, 3.8798e-25],
                                                [109.9, 3.8586999999999997e-25],
                                                [91.2, 2.20512e-25]])),
        ExplicitContinuum(j=5, i=2, **convert_alphaGrid(alphaGrid=[[820.57, 2.18e-23],
                                                [780.0, 1.8723799999999998e-23],
                                                [740.0, 1.5988399999999998e-23],
                                                [700.0, 1.3533299999999999e-23],
                                                [669.0, 1.1813799999999999e-23],
                                                [668.4, 1.1781999999999999e-23],
                                                [579.0, 7.65853e-24],
                                                [555.05, 6.74692e-24],
                                                [513.0, 5.32674e-24],
                                                [470.0, 4.09641e-24],
                                                [450.45, 3.6062e-24],
                                                [430.0, 3.1369999999999998e-24],
                                                [417.0, 2.8609999999999998e-24],
                                                [400.0, 2.52516e-24],
                                                [387.0, 2.28687e-24],
                                                [364.71, 1.9140499999999999e-24]])),
        ExplicitContinuum(j=5, i=3, **convert_alphaGrid(alphaGrid=[[1442.2, 2.91e-23],
                                                [1350.0, 2.38681e-23],
                                                [1215.0, 1.7399799999999997e-23],
                                                [1000.0, 9.700999999999998e-24],
                                                [876.0, 6.52122e-24],
                                                [820.6, 5.36057e-24]])),
        ExplicitContinuum(j=5, i=4, **convert_alphaGrid(alphaGrid=[[2239.2, 3.63e-23],
                                                [2090.0, 2.95166e-23],
                                                [1730.0, 1.67404e-23],
                                                [1670.0, 1.5058299999999998e-23],
                                                [1540.0, 1.1808399999999999e-23],
                                                [1442.3, 9.700509999999999e-24]]))
    ],
    collisions=[
        Omega(j=1, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.61, 0.613, 0.63, 0.686, 0.847, 1.24, 2.02, 3.58, 163.0, 163.0, 163.0, 163.0, 163.0]),
        Omega(j=2, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.153, 0.155, 0.162, 0.179, 0.217, 0.296, 0.444, 0.74, 31.0, 31.0, 31.0, 31.0, 31.0]),
        Omega(j=3, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.0609, 0.0618, 0.0647, 0.0713, 0.0858, 0.115, 0.169, 0.277, 11.3, 11.3, 11.3, 11.3, 11.3]),
        Omega(j=4, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.0303, 0.0308, 0.0323, 0.0355, 0.0426, 0.0569, 0.0828, 0.135, 5.43, 5.43, 5.43, 5.43, 5.43]),
        Omega(j=2, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[25.8, 28.0, 36.4, 55.0, 89.5, 143.0, 214.0, 356.0, 14900.0, 14900.0, 14900.0, 14900.0, 14900.0]),
        Omega(j=3, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[8.04, 8.69, 10.7, 14.4, 20.8, 30.0, 41.9, 65.7, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0]),
        Omega(j=4, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[3.52, 3.78, 4.54, 5.96, 8.32, 11.7, 16.1, 24.9, 924.0, 924.0, 924.0, 924.0, 924.0]),
        Omega(j=3, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[275.0, 347.0, 561.0, 923.0, 1450.0, 2120.0, 2900.0, 4460.0, 164000.0, 164000.0, 164000.0, 164000.0, 164000.0]),
        Omega(j=4, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[88.4, 104.0, 145.0, 209.0, 295.0, 398.0, 515.0, 749.0, 24700.0, 24700.0, 24700.0, 24700.0, 24700.0]),
        Omega(j=4, i=3, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[1830.0, 2500.0, 4210.0, 6680.0, 9830.0, 13500.0, 17400.0, 25200.0, 822000.0, 822000.0, 822000.0, 822000.0, 822000.0]),
        CI(j=5, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[3.06e-17, 3.3199999999999996e-17, 3.82e-17, 4.3099999999999994e-17, 4.8e-17, 5.3e-17, 5.79e-17, 6.29e-17, 1.0099999999999999e-16, 8.91e-17, 6.75e-17, 4.61e-17, 2.94e-17, 1.7799999999999998e-17, 1.0499999999999998e-17, 1.0499999999999998e-17, 1.0499999999999998e-17]),
        CI(j=5, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[1.06e-15, 9e-16, 8.99e-16, 9.46e-16, 9.56e-16, 9.07e-16, 7.68e-16, 5.68e-16, 9.819999999999998e-16, 7.36e-16, 4.69e-16, 2.84e-16, 1.6699999999999998e-16, 9.57e-17, 5.39e-17, 5.39e-17, 5.39e-17]),
        CI(j=5, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[3.8e-15, 4.32e-15, 4.73e-15, 4.68e-15, 4.25e-15, 3.5199999999999996e-15, 2.55e-15, 1.5199999999999998e-15, 3.0699999999999998e-15, 2.18e-15, 1.31e-15, 7.67e-16, 4.3799999999999997e-16, 2.4599999999999995e-16, 1.36e-16, 1.36e-16, 1.36e-16]),
        CI(j=5, i=3, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[1.44e-14, 1.5e-14, 1.48e-14, 1.3399999999999998e-14, 1.13e-14, 8.49e-15, 5.23e-15, 1.66e-15, 6.53e-15, 4.5e-15, 2.64e-15, 1.5199999999999998e-15, 8.53e-16, 4.74e-16, 2.6099999999999995e-16, 2.6099999999999995e-16, 2.6099999999999995e-16]),
        CI(j=5, i=4, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[4.06e-14, 3.86e-14, 3.42e-14, 2.87e-14, 2.25e-14, 1.58e-14, 8.57e-15, 2.3e-15, 1.1399999999999999e-14, 7.77e-15, 4.49e-15, 2.54e-15, 1.42e-15, 7.829999999999999e-16, 4.29e-16, 4.29e-16, 4.29e-16]),
        FangHRates(0,0)
    ])

    H = H_6_radyn()
    for c in H.continua:
        for i, a in enumerate(c.alphaGrid):
            c.alphaGrid[i] *= 1e2
    # for l in H.lines:
    #     l.NlambdaGen *= 2
    # H.collisions.append(FangHRates(0,0))

    reconfigure_atom(H)
    # for r in radTrans:
        # if r['atomName'] == H.name:
            # if r['transType'] == 'Line':
                # H.lines[r['lwIdx']].wavelength = r['wlGrid'].value
                # H.lines[r['lwIdx']].preserveWavelength = True

    return H

def H_6_nobb():
    H = H_6()
    H.lines = []
    reconfigure_atom(H)
    return H

def H_6_noLybbbf():
    radynQNorm = 12.85e3
    qNormRatio = radynQNorm / lw.VMICRO_CHAR
    qr = qNormRatio

    H_6_radyn = lambda: \
    AtomicModel(element=lw.PeriodicTable['H'],
    levels=[
        AtomicLevel(E=0.000000, g=2.000000, label="H I 1S 2SE", stage=0, J=Fraction(1, 2), L=0, S=Fraction(1, 2)),
        AtomicLevel(E=82257.172000, g=8.000000, label="H I 2P 2PO", stage=0, J=Fraction(7, 2), L=1, S=Fraction(1, 2)),
        AtomicLevel(E=97489.992000, g=18.000000, label="H I 3D 2DE", stage=0, J=Fraction(17, 2), L=2, S=Fraction(1, 2)),
        AtomicLevel(E=102821.219000, g=32.000000, label="H I 4F 2FO", stage=0, J=Fraction(31, 2), L=3, S=Fraction(1, 2)),
        AtomicLevel(E=105288.859000, g=50.000000, label="H I 5G 2GE", stage=0, J=Fraction(49, 2), L=4, S=Fraction(1, 2)),
        AtomicLevel(E=109754.578000, g=1.000000, label="H II continuum", stage=1, J=None, L=None, S=None),
    ],
    lines=[
        VoigtLine(j=2, i=1, f=6.414000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=100, qCore=20.000000, qWing=250.000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=5.701e+8)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=3, i=1, f=1.195000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=80, qCore=10.000000, qWing=250.0000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=5.004e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=1, f=4.471000e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=80, qCore=10.000000, qWing=250.0000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.818e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=3, i=2, f=8.431000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=10.000000, qWing=30.00000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.301e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=2, f=1.508000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=10.000000, qWing=30.00000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.115e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=3, f=1.039000e+00, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=5.000000, qWing=30.000000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.177e+07)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        # NOTE(cmo): I don't believe RADYN does both linear and quadratic Stark broadening for H, so setting Stark to these numbers seems to make the lines overly wide
        # VoigtLine(j=1, i=0, f=4.167000e-01, type=LineType.PRD, NlambdaGen=100, qCore=15.000000, qWing=600.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=2, i=0, f=7.919000e-02, type=LineType.PRD, NlambdaGen=50, qCore=10.000000, qWing=250.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=3, i=0, f=2.901000e-02, type=LineType.CRD, NlambdaGen=20, qCore=3.000000, qWing=100.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=4, i=0, f=1.395000e-02, type=LineType.CRD, NlambdaGen=20, qCore=3.000000, qWing=100.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=2, i=1, f=6.414000e-01, type=LineType.CRD, NlambdaGen=70, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=5.701e+8, stark=1.026e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=3, i=1, f=1.195000e-01, type=LineType.CRD, NlambdaGen=40, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=5.004e+08, stark=3.836e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=4, i=1, f=4.471000e-02, type=LineType.CRD, NlambdaGen=40, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=4.818e+08, stark=6.715e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=3, i=2, f=8.431000e-01, type=LineType.CRD, NlambdaGen=20, qCore=2.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=1.301e+07, stark=3.212e-4*-Const.CM_TO_M**3),
        # VoigtLine(j=4, i=2, f=1.508000e-01, type=LineType.CRD, NlambdaGen=20, qCore=2.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=1.115e+08, stark=0.000000),
        # VoigtLine(j=4, i=3, f=1.039000e+00, type=LineType.CRD, NlambdaGen=20, qCore=1.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=4.177e+07, stark=0.000000),
    ],
    continua=[
        ExplicitContinuum(j=5, i=1, **convert_alphaGrid(alphaGrid=[[364.69, 1.41e-23],
                                                [355.0, 1.3005699999999999e-23],
                                                [340.0, 1.14257e-23],
                                                [320.0, 9.52571e-24],
                                                [308.0, 8.493759999999999e-24],
                                                [299.2, 7.78632e-24],
                                                [283.0, 6.588809999999999e-24],
                                                [282.6, 6.56091e-24],
                                                [281.4, 6.47768e-24],
                                                [253.9, 4.75812e-24],
                                                [251.4, 4.61895e-24],
                                                [251.3, 4.61344e-24],
                                                [235.0, 3.77269e-24],
                                                [207.1, 2.5821899999999997e-24],
                                                [206.9, 2.5747099999999997e-24],
                                                [180.0, 1.6953699999999998e-24],
                                                [168.2, 1.3833299999999998e-24],
                                                [168.0, 1.3783999999999998e-24],
                                                [152.1, 1.0229e-24],
                                                [151.9, 1.01887e-24],
                                                [140.7, 8.097e-25],
                                                [138.9, 7.789999999999999e-25],
                                                [135.8, 7.28e-25],
                                                [133.2, 6.87e-25],
                                                [130.0, 6.3867199999999995e-25],
                                                [120.1, 5.035890000000001e-25],
                                                [119.9, 5.01078e-25],
                                                [110.1, 3.8798e-25],
                                                [109.9, 3.8586999999999997e-25],
                                                [91.2, 2.20512e-25]])),
        ExplicitContinuum(j=5, i=2, **convert_alphaGrid(alphaGrid=[[820.57, 2.18e-23],
                                                [780.0, 1.8723799999999998e-23],
                                                [740.0, 1.5988399999999998e-23],
                                                [700.0, 1.3533299999999999e-23],
                                                [669.0, 1.1813799999999999e-23],
                                                [668.4, 1.1781999999999999e-23],
                                                [579.0, 7.65853e-24],
                                                [555.05, 6.74692e-24],
                                                [513.0, 5.32674e-24],
                                                [470.0, 4.09641e-24],
                                                [450.45, 3.6062e-24],
                                                [430.0, 3.1369999999999998e-24],
                                                [417.0, 2.8609999999999998e-24],
                                                [400.0, 2.52516e-24],
                                                [387.0, 2.28687e-24],
                                                [364.71, 1.9140499999999999e-24]])),
        ExplicitContinuum(j=5, i=3, **convert_alphaGrid(alphaGrid=[[1442.2, 2.91e-23],
                                                [1350.0, 2.38681e-23],
                                                [1215.0, 1.7399799999999997e-23],
                                                [1000.0, 9.700999999999998e-24],
                                                [876.0, 6.52122e-24],
                                                [820.6, 5.36057e-24]])),
        ExplicitContinuum(j=5, i=4, **convert_alphaGrid(alphaGrid=[[2239.2, 3.63e-23],
                                                [2090.0, 2.95166e-23],
                                                [1730.0, 1.67404e-23],
                                                [1670.0, 1.5058299999999998e-23],
                                                [1540.0, 1.1808399999999999e-23],
                                                [1442.3, 9.700509999999999e-24]]))
    ],
    collisions=[
        Omega(j=1, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.61, 0.613, 0.63, 0.686, 0.847, 1.24, 2.02, 3.58, 163.0, 163.0, 163.0, 163.0, 163.0]),
        Omega(j=2, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.153, 0.155, 0.162, 0.179, 0.217, 0.296, 0.444, 0.74, 31.0, 31.0, 31.0, 31.0, 31.0]),
        Omega(j=3, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.0609, 0.0618, 0.0647, 0.0713, 0.0858, 0.115, 0.169, 0.277, 11.3, 11.3, 11.3, 11.3, 11.3]),
        Omega(j=4, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.0303, 0.0308, 0.0323, 0.0355, 0.0426, 0.0569, 0.0828, 0.135, 5.43, 5.43, 5.43, 5.43, 5.43]),
        Omega(j=2, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[25.8, 28.0, 36.4, 55.0, 89.5, 143.0, 214.0, 356.0, 14900.0, 14900.0, 14900.0, 14900.0, 14900.0]),
        Omega(j=3, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[8.04, 8.69, 10.7, 14.4, 20.8, 30.0, 41.9, 65.7, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0]),
        Omega(j=4, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[3.52, 3.78, 4.54, 5.96, 8.32, 11.7, 16.1, 24.9, 924.0, 924.0, 924.0, 924.0, 924.0]),
        Omega(j=3, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[275.0, 347.0, 561.0, 923.0, 1450.0, 2120.0, 2900.0, 4460.0, 164000.0, 164000.0, 164000.0, 164000.0, 164000.0]),
        Omega(j=4, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[88.4, 104.0, 145.0, 209.0, 295.0, 398.0, 515.0, 749.0, 24700.0, 24700.0, 24700.0, 24700.0, 24700.0]),
        Omega(j=4, i=3, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[1830.0, 2500.0, 4210.0, 6680.0, 9830.0, 13500.0, 17400.0, 25200.0, 822000.0, 822000.0, 822000.0, 822000.0, 822000.0]),
        CI(j=5, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[3.06e-17, 3.3199999999999996e-17, 3.82e-17, 4.3099999999999994e-17, 4.8e-17, 5.3e-17, 5.79e-17, 6.29e-17, 1.0099999999999999e-16, 8.91e-17, 6.75e-17, 4.61e-17, 2.94e-17, 1.7799999999999998e-17, 1.0499999999999998e-17, 1.0499999999999998e-17, 1.0499999999999998e-17]),
        CI(j=5, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[1.06e-15, 9e-16, 8.99e-16, 9.46e-16, 9.56e-16, 9.07e-16, 7.68e-16, 5.68e-16, 9.819999999999998e-16, 7.36e-16, 4.69e-16, 2.84e-16, 1.6699999999999998e-16, 9.57e-17, 5.39e-17, 5.39e-17, 5.39e-17]),
        CI(j=5, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[3.8e-15, 4.32e-15, 4.73e-15, 4.68e-15, 4.25e-15, 3.5199999999999996e-15, 2.55e-15, 1.5199999999999998e-15, 3.0699999999999998e-15, 2.18e-15, 1.31e-15, 7.67e-16, 4.3799999999999997e-16, 2.4599999999999995e-16, 1.36e-16, 1.36e-16, 1.36e-16]),
        CI(j=5, i=3, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[1.44e-14, 1.5e-14, 1.48e-14, 1.3399999999999998e-14, 1.13e-14, 8.49e-15, 5.23e-15, 1.66e-15, 6.53e-15, 4.5e-15, 2.64e-15, 1.5199999999999998e-15, 8.53e-16, 4.74e-16, 2.6099999999999995e-16, 2.6099999999999995e-16, 2.6099999999999995e-16]),
        CI(j=5, i=4, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[4.06e-14, 3.86e-14, 3.42e-14, 2.87e-14, 2.25e-14, 1.58e-14, 8.57e-15, 2.3e-15, 1.1399999999999999e-14, 7.77e-15, 4.49e-15, 2.54e-15, 1.42e-15, 7.829999999999999e-16, 4.29e-16, 4.29e-16, 4.29e-16]),
        FangHRates(0,0)
    ])

    H = H_6_radyn()
    for c in H.continua:
        for i, a in enumerate(c.alphaGrid):
            c.alphaGrid[i] *= 1e2
    # for l in H.lines:
    #     l.NlambdaGen *= 2
    # H.collisions.append(FangHRates(0,0))

    reconfigure_atom(H)
    # for r in radTrans:
        # if r['atomName'] == H.name:
            # if r['transType'] == 'Line':
                # H.lines[r['lwIdx']].wavelength = r['wlGrid'].value
                # H.lines[r['lwIdx']].preserveWavelength = True

    return H

def H_6_nasa():
    radynQNorm = 12.85e3
    qNormRatio = radynQNorm / lw.VMICRO_CHAR
    qr = qNormRatio
    H_6_radyn_nasa = lambda: \
    AtomicModel(element=lw.PeriodicTable["H"],
    levels=[
        AtomicLevel(E=0.000000, g=2.000000, label="H I 1S 2SE", stage=0, J=Fraction(1, 2), L=0, S=Fraction(1, 2)),
        AtomicLevel(E=82257.172000, g=8.000000, label="H I 2P 2PO", stage=0, J=Fraction(7, 2), L=1, S=Fraction(1, 2)),
        AtomicLevel(E=97489.992000, g=18.000000, label="H I 3D 2DE", stage=0, J=Fraction(17, 2), L=2, S=Fraction(1, 2)),
        AtomicLevel(E=102821.219000, g=32.000000, label="H I 4F 2FO", stage=0, J=Fraction(31, 2), L=3, S=Fraction(1, 2)),
        AtomicLevel(E=105288.859000, g=50.000000, label="H I 5G 2GE", stage=0, J=Fraction(49, 2), L=4, S=Fraction(1, 2)),
        AtomicLevel(E=109754.578000, g=1.000000, label="H II continuum", stage=1, J=None, L=None, S=None),
    ],
    lines=[
        VoigtLine(j=1, i=0, f=4.167000e-01, type=LineType.PRD, quadrature=LinearCoreExpWings(Nlambda=100, qCore=10.000000*qr, qWing=10.000000*qr), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.702e8)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=2, i=0, f=7.919000e-02, type=LineType.PRD, quadrature=LinearCoreExpWings(Nlambda=50, qCore=10.000000*qr, qWing=10.000000*qr),  broadening=LineBroadening(natural=[RadiativeBroadening(gamma=9.991e7)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=3, i=0, f=2.901000e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=31, qCore=10.000000*qr, qWing=10.000000*qr),  broadening=LineBroadening(natural=[RadiativeBroadening(gamma=3.021e7)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=0, f=1.395000e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=31, qCore=10.000000*qr, qWing=10.000000*qr),  broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.156e7)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=2, i=1, f=6.414000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=70, qCore=3.000000*qr, qWing=200.000000*qr),  broadening=LineBroadening(natural=[RadiativeBroadening(gamma=5.701e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=3, i=1, f=1.195000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=3.000000*qr, qWing=200.000000*qr),  broadening=LineBroadening(natural=[RadiativeBroadening(gamma=5.004e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=1, f=4.471000e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=40, qCore=3.000000*qr, qWing=400.000000*qr),  broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.818e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=3, i=2, f=8.431000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=31, qCore=2.000000*qr, qWing=100.000000*qr),  broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.301e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=2, f=1.508000e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=31, qCore=2.000000*qr, qWing=50.000000*qr),  broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.115e+08)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        VoigtLine(j=4, i=3, f=1.039000e+00, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=31, qCore=1.000000*qr, qWing=50.000000*qr),  broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.177e+07)], elastic=[VdwRadyn(vals=[1.0]), HydrogenLinearStarkBroadening()])),
        # NOTE(cmo): I don't believe RADYN does both linear and quadratic Stark broadening for H, so setting Stark to these numbers seems to make the lines overly wide
        # VoigtLine(j=1, i=0, f=4.167000e-01, type=LineType.PRD, NlambdaGen=100, qCore=15.000000, qWing=600.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=2, i=0, f=7.919000e-02, type=LineType.PRD, NlambdaGen=50, qCore=10.000000, qWing=250.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=3, i=0, f=2.901000e-02, type=LineType.CRD, NlambdaGen=20, qCore=3.000000, qWing=100.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=4, i=0, f=1.395000e-02, type=LineType.CRD, NlambdaGen=20, qCore=3.000000, qWing=100.000000, vdw=VdwUnsold(vals=[0.0, 0.0]), gRad=0.0, stark=0.0),
        # VoigtLine(j=2, i=1, f=6.414000e-01, type=LineType.CRD, NlambdaGen=70, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=5.701e+8, stark=1.026e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=3, i=1, f=1.195000e-01, type=LineType.CRD, NlambdaGen=40, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=5.004e+08, stark=3.836e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=4, i=1, f=4.471000e-02, type=LineType.CRD, NlambdaGen=40, qCore=3.000000, qWing=250.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=4.818e+08, stark=6.715e-2*-Const.CM_TO_M**3),
        # VoigtLine(j=3, i=2, f=8.431000e-01, type=LineType.CRD, NlambdaGen=20, qCore=2.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=1.301e+07, stark=3.212e-4*-Const.CM_TO_M**3),
        # VoigtLine(j=4, i=2, f=1.508000e-01, type=LineType.CRD, NlambdaGen=20, qCore=2.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=1.115e+08, stark=0.000000),
        # VoigtLine(j=4, i=3, f=1.039000e+00, type=LineType.CRD, NlambdaGen=20, qCore=1.000000, qWing=30.000000, vdw=VdwUnsold(vals=[1.0, 1.0]), gRad=4.177e+07, stark=0.000000),
    ],
    continua=[
        ExplicitContinuum(j=5, i=0, **convert_alphaGrid(alphaGrid=[[91.17, 6.538849999999999e-24],
                                                [85.0, 5.299099999999999e-24],
                                                [80.0, 4.41789e-24],
                                                [75.0, 3.64023e-24],
                                                [70.0, 2.95964e-24],
                                                [60.0, 1.8638e-24],
                                                [50.5, 1.1112699999999999e-24]])),
        ExplicitContinuum(j=5, i=1, **convert_alphaGrid(alphaGrid=[[364.69, 1.41e-23],
                                                [355.0, 1.3005699999999999e-23],
                                                [340.0, 1.14257e-23],
                                                [320.0, 9.52571e-24],
                                                [308.0, 8.493759999999999e-24],
                                                [299.2, 7.78632e-24],
                                                [283.0, 6.588809999999999e-24],
                                                [282.6, 6.56091e-24],
                                                [281.4, 6.47768e-24],
                                                [253.9, 4.75812e-24],
                                                [251.4, 4.61895e-24],
                                                [251.3, 4.61344e-24],
                                                [235.0, 3.77269e-24],
                                                [207.1, 2.5821899999999997e-24],
                                                [206.9, 2.5747099999999997e-24],
                                                [180.0, 1.6953699999999998e-24],
                                                [168.2, 1.3833299999999998e-24],
                                                [168.0, 1.3783999999999998e-24],
                                                [152.1, 1.0229e-24],
                                                [151.9, 1.01887e-24],
                                                [140.7, 8.097e-25],
                                                [138.9, 7.789999999999999e-25],
                                                [135.8, 7.28e-25],
                                                [133.2, 6.87e-25],
                                                [130.0, 6.3867199999999995e-25],
                                                [120.1, 5.035890000000001e-25],
                                                [119.9, 5.01078e-25],
                                                [110.1, 3.8798e-25],
                                                [109.9, 3.8586999999999997e-25],
                                                [91.2, 2.20512e-25]])),
        ExplicitContinuum(j=5, i=2, **convert_alphaGrid(alphaGrid=[[820.57, 2.18e-23],
                                                [780.0, 1.8723799999999998e-23],
                                                [740.0, 1.5988399999999998e-23],
                                                [700.0, 1.3533299999999999e-23],
                                                [669.0, 1.1813799999999999e-23],
                                                [668.4, 1.1781999999999999e-23],
                                                [579.0, 7.65853e-24],
                                                [555.05, 6.74692e-24],
                                                [513.0, 5.32674e-24],
                                                [470.0, 4.09641e-24],
                                                [450.45, 3.6062e-24],
                                                [430.0, 3.1369999999999998e-24],
                                                [417.0, 2.8609999999999998e-24],
                                                [400.0, 2.52516e-24],
                                                [387.0, 2.28687e-24],
                                                [364.71, 1.9140499999999999e-24]])),
        ExplicitContinuum(j=5, i=3, **convert_alphaGrid(alphaGrid=[[1442.2, 2.91e-23],
                                                [1350.0, 2.38681e-23],
                                                [1215.0, 1.7399799999999997e-23],
                                                [1000.0, 9.700999999999998e-24],
                                                [876.0, 6.52122e-24],
                                                [820.6, 5.36057e-24]])),
        ExplicitContinuum(j=5, i=4, **convert_alphaGrid(alphaGrid=[[2239.2, 3.63e-23],
                                                [2090.0, 2.95166e-23],
                                                [1730.0, 1.67404e-23],
                                                [1670.0, 1.5058299999999998e-23],
                                                [1540.0, 1.1808399999999999e-23],
                                                [1442.3, 9.700509999999999e-24]]))
    ],
    collisions=[
        Omega(j=1, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.61, 0.613, 0.63, 0.686, 0.847, 1.24, 2.02, 3.58, 163.0, 163.0, 163.0, 163.0, 163.0]),
        Omega(j=2, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.153, 0.155, 0.162, 0.179, 0.217, 0.296, 0.444, 0.74, 31.0, 31.0, 31.0, 31.0, 31.0]),
        Omega(j=3, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.0609, 0.0618, 0.0647, 0.0713, 0.0858, 0.115, 0.169, 0.277, 11.3, 11.3, 11.3, 11.3, 11.3]),
        Omega(j=4, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[0.0303, 0.0308, 0.0323, 0.0355, 0.0426, 0.0569, 0.0828, 0.135, 5.43, 5.43, 5.43, 5.43, 5.43]),
        Omega(j=2, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[25.8, 28.0, 36.4, 55.0, 89.5, 143.0, 214.0, 356.0, 14900.0, 14900.0, 14900.0, 14900.0, 14900.0]),
        Omega(j=3, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[8.04, 8.69, 10.7, 14.4, 20.8, 30.0, 41.9, 65.7, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0]),
        Omega(j=4, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[3.52, 3.78, 4.54, 5.96, 8.32, 11.7, 16.1, 24.9, 924.0, 924.0, 924.0, 924.0, 924.0]),
        Omega(j=3, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[275.0, 347.0, 561.0, 923.0, 1450.0, 2120.0, 2900.0, 4460.0, 164000.0, 164000.0, 164000.0, 164000.0, 164000.0]),
        Omega(j=4, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[88.4, 104.0, 145.0, 209.0, 295.0, 398.0, 515.0, 749.0, 24700.0, 24700.0, 24700.0, 24700.0, 24700.0]),
        Omega(j=4, i=3, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 2000000.0, 5000000.0, 10000000.0, 30000000.0, 50000000.0], rates=[1830.0, 2500.0, 4210.0, 6680.0, 9830.0, 13500.0, 17400.0, 25200.0, 822000.0, 822000.0, 822000.0, 822000.0, 822000.0]),
        CI(j=5, i=0, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[3.06e-17, 3.3199999999999996e-17, 3.82e-17, 4.3099999999999994e-17, 4.8e-17, 5.3e-17, 5.79e-17, 6.29e-17, 1.0099999999999999e-16, 8.91e-17, 6.75e-17, 4.61e-17, 2.94e-17, 1.7799999999999998e-17, 1.0499999999999998e-17, 1.0499999999999998e-17, 1.0499999999999998e-17]),
        CI(j=5, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[1.06e-15, 9e-16, 8.99e-16, 9.46e-16, 9.56e-16, 9.07e-16, 7.68e-16, 5.68e-16, 9.819999999999998e-16, 7.36e-16, 4.69e-16, 2.84e-16, 1.6699999999999998e-16, 9.57e-17, 5.39e-17, 5.39e-17, 5.39e-17]),
        CI(j=5, i=2, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[3.8e-15, 4.32e-15, 4.73e-15, 4.68e-15, 4.25e-15, 3.5199999999999996e-15, 2.55e-15, 1.5199999999999998e-15, 3.0699999999999998e-15, 2.18e-15, 1.31e-15, 7.67e-16, 4.3799999999999997e-16, 2.4599999999999995e-16, 1.36e-16, 1.36e-16, 1.36e-16]),
        CI(j=5, i=3, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[1.44e-14, 1.5e-14, 1.48e-14, 1.3399999999999998e-14, 1.13e-14, 8.49e-15, 5.23e-15, 1.66e-15, 6.53e-15, 4.5e-15, 2.64e-15, 1.5199999999999998e-15, 8.53e-16, 4.74e-16, 2.6099999999999995e-16, 2.6099999999999995e-16, 2.6099999999999995e-16]),
        CI(j=5, i=4, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 300000.0, 500000.0, 1000000.0, 2000000.0, 4000000.0, 8000000.0, 16000000.0, 30000000.0, 50000000.0], rates=[4.06e-14, 3.86e-14, 3.42e-14, 2.87e-14, 2.25e-14, 1.58e-14, 8.57e-15, 2.3e-15, 1.1399999999999999e-14, 7.77e-15, 4.49e-15, 2.54e-15, 1.42e-15, 7.829999999999999e-16, 4.29e-16, 4.29e-16, 4.29e-16]),
        FangHRates(0,0)
    ])

    H = H_6_radyn_nasa()
    for c in H.continua:
        for i, a in enumerate(c.alphaGrid):
            c.alphaGrid[i] *= 1e2
    # for l in H.lines:
    #     l.NlambdaGen *= 2
    # H.collisions.append(FangHRates(0,0))
    # for c in H.collisions:
    #     if isinstance(c, Omega):
    #         c.rates = [r * 1e-6 for r in c.rates]

    reconfigure_atom(H)
    # for r in radTrans:
        # if r['atomName'] == H.name:
            # if r['transType'] == 'Line':
                # H.lines[r['lwIdx']].wavelength = r['wlGrid'].value
                # H.lines[r['lwIdx']].preserveWavelength = True

    return H

@dataclass
class Ar85CeaCaII(CollisionalRates):
    fudge: float = 1.0

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.iLevel = atom.levels[self.i]
        self.jLevel = atom.levels[self.j]

    def __repr__(self):
        s = 'Ar85CeaCaII(j=%d, i=%d, fudge=%e)' % (self.j, self.i, self.fudge)
        return s

    def compute_rates(self, atmos, eqPops, Cmat):
        # CaII is in the K isoelectronic sequence
        zz = 20
        kBT = Const.KBoltzmann * atmos.temperature / Const.EV
        # NOTE(cmo): CaII has a special case we apply directly
        # TODO(cmo): Which bits are the branching ratios?! -- Read AR85 in more detail
        a = 6.0e-17 # NOTE(cmo): From looking at the AR85 paper, (page 430), should this instead be 9.8e-17 (Ca+)
        iea = 25.0
        # NOTE(cmo): From Appendix A to AR85 for Ca+
        # a = 9.8e-17
        # iea = 29.0
        # NOTE(cmo): Changed above back for consistency, need to look into which is technically more correct though
        y = iea / kBT
        f1y = fone(y)
        b = 1.12
        cUp = 6.69e7 * a * iea / np.sqrt(kBT) * np.exp(-y)*(1.0 + b*f1y)
        # NOTE(cmo): Rates are in cm-3 s-1, so use ne in cm-3
        cUp *= self.fudge * atmos.ne * Const.CM_TO_M**3
        Cmat[self.j, self.i, :] += cUp


@dataclass
class Shull82(CollisionalRates):
    row: int
    col: int
    #  = 3, 1 for the Ca rates in Radyn
    aCol: float
    tCol: float
    aRad: float
    xRad: float
    aDi: float
    bDi: float
    t0: float
    t1: float

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.iLevel = atom.levels[self.i]
        self.jLevel = atom.levels[self.j]

    def __repr__(self):
        s = 'Shull82(row=%d, col=%d, aCol=%e, tCol=%e, aRad=%e, xRad=%e, aDi=%e, bDi=%e, t0=%e, t1=%e)' % (self.row, self.col, self.aCol, self.tCol, self.aRad, self.xRad, self.aDi, self.bDi, self.t0, self.t1)
        return s

    def compute_rates(self, atmos, eqPops, Cmat):
        nstar = eqPops.atomicPops[self.atom.element].nStar
        # NOTE(cmo): Summers direct recombination rates
        zz = self.jLevel.stage
        rhoq = (atmos.ne * Const.CM_TO_M**3) / zz**7
        x = (0.5 * zz + (self.col - 1)) * self.row / 3
        beta = -0.2 / np.log(x + np.e)

        tg = atmos.temperature
        # NOTE(cmo): This is the RH formulation
        # rho0 = 30.0 + 50.0*x
        # y = (1.0 + rhoq/rho0)**beta
        # summersScaling = 1.0
        # summers = summersScaling * y + (1.0 - summersScaling)
        # NOTE(cmo): This is a direct application of RADYN's method (for CaII)
        rho0 = 30
        summers = 1.0 / (1.0 + rhoq / rho0)**0.14


        cDown = self.aRad * (tg * 1e-4)**(-self.xRad)
        cDown += summers * self.aDi / tg / np.sqrt(tg) * np.exp(-self.t0 / tg) * (1.0 + self.bDi * np.exp(-self.t1 / tg))

        cUp = self.aCol * np.sqrt(tg) * np.exp(-self.tCol / tg) / (1.0 + 0.1 * tg / self.tCol)

        # NOTE(cmo): Rates are in cm-3 s-1, so use ne in cm-3
        cDown *= atmos.ne * Const.CM_TO_M**3
        cUp *= atmos.ne * Const.CM_TO_M**3

        # NOTE(cmo): 3-body recombination (high density limit)
        cDown += cUp * nstar[self.i, :] / nstar[self.j, :]

        Cmat[self.i, self.j, :] += cDown
        Cmat[self.j, self.i, :] += cUp

def CaII():
    CaII_radyn = lambda: \
    AtomicModel(element=lw.PeriodicTable['Ca'],
    levels=[
        AtomicLevel(E=0.000000, g=2.000000, label="CA II 3P6 4S 2SE", stage=1, J=Fraction(1, 2), L=0, S=Fraction(1, 2)),
        AtomicLevel(E=13650.248000, g=4.000000, label="CA II 3P6 3D 2DE 3", stage=1, J=Fraction(3, 2), L=2, S=Fraction(1, 2)),
        AtomicLevel(E=13710.90000, g=6.000000, label="CA II 3P6 3D 2DE 5", stage=1, J=Fraction(5, 2), L=2, S=Fraction(1, 2)),
        AtomicLevel(E=25191.535000, g=2.000000, label="CA II 3P6 4P 2PO 1", stage=1, J=Fraction(1, 2), L=1, S=Fraction(1, 2)),
        AtomicLevel(E=25414.465000, g=4.000000, label="CA II 3P6 4P 2PO 3", stage=1, J=Fraction(3, 2), L=1, S=Fraction(1, 2)),
        AtomicLevel(E=95751.870000, g=1.000000, label="CA III 3P6 1SE", stage=2, J=Fraction(0, 1), L=0, S=Fraction(0, 1)),
    ],
    lines=[
        VoigtLine(j=3, i=0, f=3.16e-01, type=LineType.PRD, quadrature=LinearCoreExpWings(Nlambda= 80, qCore=30.000000, qWing=1500.00000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.42e+08)], elastic=[MultiplicativeStarkBroadening(5.458e-7*Const.CM_TO_M**3), VdwRadyn(vals=[1.62])])),
        VoigtLine(j=4, i=0, f=6.37e-01, type=LineType.PRD, quadrature=LinearCoreExpWings(Nlambda= 80, qCore=30.000000, qWing=1500.00000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.46e+08)], elastic=[MultiplicativeStarkBroadening(5.410e-7*Const.CM_TO_M**3), VdwRadyn(vals=[1.61])])),
        VoigtLine(j=3, i=1, f=4.73e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda= 80, qCore=10.000000, qWing=200.000000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.42e+08)], elastic=[MultiplicativeStarkBroadening(2.673e-7*Const.CM_TO_M**3), VdwRadyn(vals=[2.04])])),
        VoigtLine(j=4, i=1, f=9.60e-03, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda= 80, qCore=10.000000, qWing=150.000000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.46e+08)], elastic=[MultiplicativeStarkBroadening(3.000e-6*Const.CM_TO_M**3), VdwRadyn(vals=[2.01])])),
        VoigtLine(j=4, i=2, f=5.74e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=100, qCore=10.000000, qWing=200.000000), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.46e+08)], elastic=[MultiplicativeStarkBroadening(3.000e-6*Const.CM_TO_M**3), VdwRadyn(vals=[2.01])])),
    ],
    continua=[
        ExplicitContinuum(j=5, i=0, **convert_alphaGrid(alphaGrid=[[103.7, 3.74e-25],
                                                [101.0, 3.46e-25],
                                                [99.0, 3.26e-25],
                                                [97.256, 3.09e-25],
                                                [94.977, 2.8799999999999997e-25],
                                                [91.2, 2.5499999999999997e-25],
                                                [91.17, 2.54e-25],
                                                [90.0, 2.44e-25],
                                                [88.0, 2.28e-25],
                                                [85.0, 2.06e-25],
                                                [84.0, 1.99e-25],
                                                [82.46000000000001, 1.88e-25],
                                                [80.7, 1.7599999999999998e-25],
                                                [78.55, 1.62e-25],
                                                [73.6, 1.3399999999999999e-25],
                                                [70.0, 1.15e-25],
                                                [65.0, 9.22e-26],
                                                [60.36, 7.379999999999999e-26],
                                                [56.17999999999999, 5.95e-26],
                                                [51.589999999999996, 4.61e-26],
                                                [49.010000000000005, 3.95e-26],
                                                [45.31, 3.12e-26],
                                                [42.85, 2.64e-26],
                                                [39.88, 2.13e-26]])),
        ExplicitContinuum(j=5, i=1, **convert_alphaGrid(alphaGrid=[[121.56700000000001, 6.819999999999999e-24],
                                                [120.1, 6.7e-24],
                                                [119.9, 6.68e-24],
                                                [118.0, 6.5299999999999995e-24],
                                                [116.85999999999999, 6.429999999999999e-24],
                                                [115.0, 6.2699999999999995e-24],
                                                [114.0, 6.19e-24],
                                                [112.0, 6.02e-24],
                                                [110.1, 5.8599999999999996e-24],
                                                [110.093, 5.8599999999999996e-24],
                                                [110.016, 5.849999999999999e-24],
                                                [108.0, 5.679999999999999e-24],
                                                [106.0, 5.5099999999999995e-24],
                                                [102.572, 5.22e-24],
                                                [99.0, 4.9e-24],
                                                [94.977, 4.549999999999999e-24],
                                                [91.2, 4.2099999999999995e-24],
                                                [91.17, 4.2099999999999995e-24],
                                                [90.0, 4.0999999999999994e-24],
                                                [88.0, 3.92e-24],
                                                [85.0, 3.64e-24],
                                                [82.46000000000001, 3.4099999999999995e-24],
                                                [80.0, 3.1899999999999998e-24],
                                                [76.2, 2.8499999999999996e-24],
                                                [70.0, 2.2799999999999998e-24],
                                                [65.0, 1.81e-24],
                                                [60.36, 1.32e-24],
                                                [56.17999999999999, 8.32e-25],
                                                [51.589999999999996, 4.63e-25],
                                                [49.010000000000005, 3.98e-25],
                                                [45.31, 3.15e-25],
                                                [42.85, 2.6600000000000003e-25]])),
        ExplicitContinuum(j=5, i=2, **convert_alphaGrid(alphaGrid=[[121.56700000000001, 6.819999999999999e-24],
                                                [120.1, 6.7e-24],
                                                [119.9, 6.68e-24],
                                                [118.0, 6.5299999999999995e-24],
                                                [116.85999999999999, 6.429999999999999e-24],
                                                [115.0, 6.2699999999999995e-24],
                                                [114.0, 6.19e-24],
                                                [112.0, 6.02e-24],
                                                [110.1, 5.8599999999999996e-24],
                                                [110.093, 5.8599999999999996e-24],
                                                [110.016, 5.849999999999999e-24],
                                                [108.0, 5.679999999999999e-24],
                                                [106.0, 5.5099999999999995e-24],
                                                [102.572, 5.22e-24],
                                                [99.0, 4.9e-24],
                                                [94.977, 4.549999999999999e-24],
                                                [91.2, 4.2099999999999995e-24],
                                                [91.17, 4.2099999999999995e-24],
                                                [90.0, 4.0999999999999994e-24],
                                                [88.0, 3.92e-24],
                                                [85.0, 3.64e-24],
                                                [82.46000000000001, 3.4099999999999995e-24],
                                                [80.0, 3.1899999999999998e-24],
                                                [76.2, 2.8499999999999996e-24],
                                                [70.0, 2.2799999999999998e-24],
                                                [65.0, 1.81e-24],
                                                [60.36, 1.32e-24],
                                                [56.17999999999999, 8.32e-25],
                                                [51.589999999999996, 4.63e-25],
                                                [49.010000000000005, 3.98e-25],
                                                [45.31, 3.15e-25],
                                                [42.85, 2.6600000000000003e-25]])),
        ExplicitContinuum(j=5, i=3, **convert_alphaGrid(alphaGrid=[[140.277, 2.94e-24],
                                                [139.375, 2.87e-24],
                                                [138.0, 2.77e-24],
                                                [136.0, 2.64e-24],
                                                [133.5, 2.48e-24],
                                                [132.0, 2.38e-24],
                                                [129.45499999999998, 2.23e-24],
                                                [126.5, 2.0499999999999997e-24],
                                                [124.0, 1.9099999999999997e-24],
                                                [123.978, 1.9099999999999997e-24],
                                                [123.881, 1.8999999999999998e-24],
                                                [120.1, 1.7e-24],
                                                [119.9, 1.69e-24],
                                                [116.0, 1.4999999999999998e-24],
                                                [110.1, 1.24e-24],
                                                [110.093, 1.24e-24],
                                                [110.016, 1.24e-24],
                                                [104.42, 1.0099999999999999e-24],
                                                [97.256, 7.72e-25],
                                                [91.2, 6.029999999999999e-25],
                                                [91.17, 6.0199999999999995e-25],
                                                [88.0, 5.27e-25],
                                                [85.0, 4.63e-25],
                                                [80.0, 3.74e-25],
                                                [73.6, 2.89e-25],
                                                [67.61, 2.38e-25],
                                                [62.55800000000001, 2.17e-25],
                                                [58.435, 1.9399999999999998e-25],
                                                [56.17999999999999, 1.73e-25],
                                                [51.589999999999996, 1.3399999999999999e-25],
                                                [49.010000000000005, 1.14e-25],
                                                [45.31, 9.05e-26],
                                                [42.85, 7.66e-26]])),
        ExplicitContinuum(j=5, i=4, **convert_alphaGrid(alphaGrid=[[142.0, 3.0599999999999998e-24],
                                                [140.277, 2.94e-24],
                                                [138.0, 2.77e-24],
                                                [136.0, 2.64e-24],
                                                [133.5, 2.48e-24],
                                                [130.927, 2.31e-24],
                                                [129.45499999999998, 2.23e-24],
                                                [126.5, 2.0499999999999997e-24],
                                                [124.0, 1.9099999999999997e-24],
                                                [123.978, 1.9099999999999997e-24],
                                                [123.881, 1.8999999999999998e-24],
                                                [120.1, 1.7e-24],
                                                [119.9, 1.69e-24],
                                                [116.0, 1.4999999999999998e-24],
                                                [110.1, 1.24e-24],
                                                [110.093, 1.24e-24],
                                                [110.016, 1.24e-24],
                                                [104.42, 1.0099999999999999e-24],
                                                [97.256, 7.72e-25],
                                                [91.2, 6.029999999999999e-25],
                                                [91.17, 6.0199999999999995e-25],
                                                [88.0, 5.27e-25],
                                                [85.0, 4.63e-25],
                                                [80.0, 3.74e-25],
                                                [73.6, 2.89e-25],
                                                [67.61, 2.38e-25],
                                                [62.55800000000001, 2.17e-25],
                                                [58.435, 1.9399999999999998e-25],
                                                [56.17999999999999, 1.73e-25],
                                                [51.589999999999996, 1.3399999999999999e-25],
                                                [49.010000000000005, 1.14e-25],
                                                [45.31, 9.05e-26],
                                                [42.85, 7.66e-26]]))
    ],
    collisions=[
        Omega(j=0, i=1, temperature=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 10000.0, 15000.0, 18000.0, 20000.0, 25000.0, 28000.0, 30000.0, 35000.0, 38000.0, 50000.0, 100000.0, 200000.0, 400000.0, 800000.0, 1600000.0, 3200000.0, 6400000.0, 13000000.0], rates=[5.2, 5.1, 4.91, 4.58, 4.24, 4.05, 3.62, 3.43, 3.31, 3.07, 2.94, 2.85, 2.68, 2.61, 2.39, 1.94, 1.66, 1.5, 1.42, 1.37, 1.35, 1.34, 1.33]),
        Omega(j=0, i=2, temperature=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 10000.0, 15000.0, 18000.0, 20000.0, 25000.0, 28000.0, 30000.0, 35000.0, 38000.0, 50000.0, 100000.0, 200000.0, 400000.0, 800000.0, 1600000.0, 3200000.0, 6400000.0, 13000000.0], rates=[7.65, 7.51, 7.25, 6.79, 6.31, 6.02, 5.4, 5.11, 4.94, 4.56, 4.39, 4.25, 4.01, 3.88, 3.49, 2.7, 2.19, 1.9, 1.75, 1.67, 1.63, 1.61, 1.6]),
        Omega(j=0, i=3, temperature=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 10000.0, 15000.0, 18000.0, 20000.0, 25000.0, 28000.0, 30000.0, 35000.0, 38000.0, 50000.0, 100000.0, 200000.0, 400000.0, 800000.0, 1600000.0, 3200000.0, 6400000.0, 13000000.0], rates=[4.44, 4.53, 4.71, 5.01, 5.46, 5.74, 6.32, 6.59, 6.75, 7.05, 7.16, 7.22, 7.29, 7.3, 7.24, 7.53, 8.38, 9.66, 11.2, 13.0, 14.8, 16.6, 18.6]),
        Omega(j=0, i=4, temperature=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 10000.0, 15000.0, 18000.0, 20000.0, 25000.0, 28000.0, 30000.0, 35000.0, 38000.0, 50000.0, 100000.0, 200000.0, 400000.0, 800000.0, 1600000.0, 3200000.0, 6400000.0, 13000000.0], rates=[8.8, 9.01, 9.39, 10.0, 11.0, 11.5, 12.7, 13.2, 13.5, 14.1, 14.3, 14.4, 14.5, 14.5, 14.4, 15.0, 16.7, 19.2, 22.3, 25.8, 29.4, 33.1, 37.0]),
        Omega(j=1, i=2, temperature=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 10000.0, 15000.0, 18000.0, 20000.0, 25000.0, 28000.0, 30000.0, 35000.0, 38000.0, 50000.0, 100000.0, 200000.0, 400000.0, 800000.0, 1600000.0, 3200000.0, 6400000.0, 13000000.0], rates=[16.2, 15.9, 15.6, 15.3, 15.1, 15.0, 14.4, 14.0, 13.8, 13.1, 12.8, 12.5, 11.9, 11.6, 10.6, 9.11, 8.35, 7.97, 7.78, 7.69, 7.64, 7.62, 7.6]),
        Omega(j=1, i=3, temperature=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 10000.0, 15000.0, 18000.0, 20000.0, 25000.0, 28000.0, 30000.0, 35000.0, 38000.0, 50000.0, 100000.0, 200000.0, 400000.0, 800000.0, 1600000.0, 3200000.0, 6400000.0, 13000000.0], rates=[10.8, 11.0, 11.3, 11.9, 12.6, 13.0, 13.8, 14.1, 14.3, 14.6, 14.7, 14.8, 14.8, 14.7, 14.4, 14.1, 14.6, 15.5, 16.5, 17.7, 18.9, 20.1, 21.4]),
        Omega(j=1, i=4, temperature=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 10000.0, 15000.0, 18000.0, 20000.0, 25000.0, 28000.0, 30000.0, 35000.0, 38000.0, 50000.0, 100000.0, 200000.0, 400000.0, 800000.0, 1600000.0, 3200000.0, 6400000.0, 13000000.0], rates=[4.65, 4.71, 4.82, 5.04, 5.31, 5.43, 5.6, 5.63, 5.64, 5.6, 5.56, 5.52, 5.41, 5.34, 5.13, 4.88, 4.87, 5.0, 5.19, 5.41, 5.65, 5.89, 6.14]),
        Omega(j=2, i=3, temperature=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 10000.0, 15000.0, 18000.0, 20000.0, 25000.0, 28000.0, 30000.0, 35000.0, 38000.0, 50000.0, 100000.0, 200000.0, 400000.0, 800000.0, 1600000.0, 3200000.0, 6400000.0, 13000000.0], rates=[2.42, 2.41, 2.39, 2.36, 2.4, 2.42, 2.41, 2.39, 2.36, 2.28, 2.23, 2.18, 2.1, 2.05, 1.89, 1.59, 1.4, 1.3, 1.24, 1.21, 1.2, 1.19, 1.19]),
        Omega(j=2, i=4, temperature=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 10000.0, 15000.0, 18000.0, 20000.0, 25000.0, 28000.0, 30000.0, 35000.0, 38000.0, 50000.0, 100000.0, 200000.0, 400000.0, 800000.0, 1600000.0, 3200000.0, 6400000.0, 13000000.0], rates=[21.0, 21.4, 22.0, 23.2, 24.7, 25.5, 26.9, 27.5, 27.8, 28.3, 28.4, 28.5, 28.4, 28.3, 27.5, 26.9, 27.7, 29.2, 31.1, 33.2, 35.3, 37.5, 39.8]),
        Omega(j=3, i=4, temperature=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 10000.0, 15000.0, 18000.0, 20000.0, 25000.0, 28000.0, 30000.0, 35000.0, 38000.0, 50000.0, 100000.0, 200000.0, 400000.0, 800000.0, 1600000.0, 3200000.0, 6400000.0, 13000000.0], rates=[5.24, 5.26, 5.27, 5.28, 5.37, 5.43, 5.52, 5.53, 5.53, 5.5, 5.46, 5.44, 5.36, 5.3, 5.13, 4.87, 4.73, 4.67, 4.63, 4.62, 4.61, 4.61, 4.6]),

        Shull82(j=5, i=0, row=3, col=1, aCol=0.0, tCol=1.38e5, aRad=0.0, xRad=8.0e-1, aDi=5.84e-2, bDi=1.1e-1, t0=3.85e5, t1=2.45e5),

        Ar85Cdi(j=5, i=0, cdi=[[11.90, 7.90, -2.00, 0.20, -6.00],
                               [37.00, 74.30, -24.20, 7.00, -63.00],
                               [45.20, 17.60, -3.80, 1.90, -13.8]]),
        Ar85CeaCaII(j=5, i=0),
        Burgess(j=5, i=1),
        Burgess(j=5, i=2),
        Burgess(j=5, i=3),
        Burgess(j=5, i=4)
    ])

    Ca = CaII_radyn()
    for c in Ca.continua:
        for i, a in enumerate(c.alphaGrid):
            c.alphaGrid[i] *= 1e2
    # for l in Ca.lines:
    #     l.NlambdaGen *= 2

    reconfigure_atom(Ca)
    # for r in radTrans:
        # if r['atomName'] == Ca.name:
            # if r['transType'] == 'Line':
                # Ca.lines[r['lwIdx']].wavelength = r['wlGrid'].value
                # Ca.lines[r['lwIdx']].preserveWavelength = True
    return Ca


@dataclass
class CH(TemperatureInterpolationRates):
    """Collisions with Neutral Hydrogen.
    """

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom

    def __repr__(self):
        s =  'CH(j=%d, i=%d, temperature=%s, rates=%s)' % (self.j, self.i, repr(self.temperature), repr(self.rates))
        return s

    def compute_rates(self, atmos, eqPops, Cmat):
        nStar = eqPops.atomicPops[self.atom.element].nStar
        C = weno4(atmos.temperature, self.temperature, self.rates)

        Cup = atmos.hPops[0, :] * C[:]
        Cmat[self.j, self.i, :] += Cup
        Cmat[self.i, self.j, :] += Cup * nStar[self.i] / nStar[self.j]

def CaII_nasa():
    radynQNorm = 12.85e3
    qNormRatio = radynQNorm / lw.VMICRO_CHAR
    qr = qNormRatio

    tempGrid = [1000.0, 5e7]
    CaII_radyn_nasa = lambda: \
    AtomicModel(element=lw.PeriodicTable["Ca"],
    levels=[
        AtomicLevel(E=0.000000, g=2.000000, label="CA II 3P6 4S 2SE", stage=1, J=Fraction(1, 2), L=0, S=Fraction(1, 2)),
        AtomicLevel(E=13650.248000, g=4.000000, label="CA II 3P6 3D 2DE 3", stage=1, J=Fraction(3, 2), L=2, S=Fraction(1, 2)),
        AtomicLevel(E=13710.90000, g=6.000000, label="CA II 3P6 3D 2DE 5", stage=1, J=Fraction(5, 2), L=2, S=Fraction(1, 2)),
        AtomicLevel(E=25191.535000, g=2.000000, label="CA II 3P6 4P 2PO 1", stage=1, J=Fraction(1, 2), L=1, S=Fraction(1, 2)),
        AtomicLevel(E=25414.465000, g=4.000000, label="CA II 3P6 4P 2PO 3", stage=1, J=Fraction(3, 2), L=1, S=Fraction(1, 2)),
        AtomicLevel(E=95785.470000, g=1.000000, label="CA III 3P6 1SE", stage=2, J=Fraction(0, 1), L=0, S=Fraction(0, 1)),
    ],
    lines=[
        VoigtLine(j=3, i=0, f=3.30e-01, type=LineType.PRD, quadrature=LinearCoreExpWings(Nlambda=101, qCore=3.0*qr, qWing=300.0*qr), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.48e+08)], elastic=[MultiplicativeStarkBroadening(3.0e-6*Const.CM_TO_M**3), VdwRadyn(vals=[.62])])),
        VoigtLine(j=4, i=0, f=6.60e-01, type=LineType.PRD, quadrature=LinearCoreExpWings(Nlambda=101, qCore=3.0*qr, qWing=300.0*qr), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.50e+08)], elastic=[MultiplicativeStarkBroadening(3.0e-6*Const.CM_TO_M**3), VdwRadyn(vals=[1.61])])),
        VoigtLine(j=3, i=1, f=4.42e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=101, qCore=1.0*qr, qWing=150.0*qr), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.48e+08)], elastic=[MultiplicativeStarkBroadening(3.0e-6*Const.CM_TO_M**3), VdwRadyn(vals=[2.04])])),
        VoigtLine(j=4, i=1, f=8.83e-03, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=101, qCore=1.0*qr, qWing=150.0*qr), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.50e+08)], elastic=[MultiplicativeStarkBroadening(3.0e-6*Const.CM_TO_M**3), VdwRadyn(vals=[2.01])])),
        VoigtLine(j=4, i=2, f=5.30e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(Nlambda=101, qCore=1.0*qr, qWing=150.0*qr), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.50e+08)], elastic=[MultiplicativeStarkBroadening(3.0e-6*Const.CM_TO_M**3), VdwRadyn(vals=[2.01])])),
    ],
    continua=[
        ExplicitContinuum(j=5, i=0, **convert_alphaGrid(alphaGrid=[[104.42, 2.036e-23],
                                               [91.2, 2.1400000000000003e-23],
                                               [91.17, 2.1400000000000003e-23],
                                               [85.0, 2.172e-23],
                                               [75.0, 2.103e-23],
                                               [60.0, 1.82e-23]])),
        ExplicitContinuum(j=5, i=1, **convert_alphaGrid(alphaGrid=[[121.84, 6.148400000000001e-22],
                                               [115.0, 5.9073e-22],
                                               [104.42, 5.5114e-22],
                                               [91.2, 4.8267999999999995e-22],
                                               [91.17, 4.8267999999999995e-22],
                                               [80.0, 4.315e-22],
                                               [70.0, 3.7619e-22],
                                               [60.0, 3.1682e-22]])),
        ExplicitContinuum(j=5, i=2, **convert_alphaGrid(alphaGrid=[[121.84, 6.148400000000001e-22],
                                               [115.0, 5.9073e-22],
                                               [104.42, 5.5114e-22],
                                               [91.2, 4.8267999999999995e-22],
                                               [91.17, 4.8267999999999995e-22],
                                               [80.0, 4.315e-22],
                                               [70.0, 3.7619e-22],
                                               [60.0, 3.1682e-22]])),
        ExplicitContinuum(j=5, i=3, **convert_alphaGrid(alphaGrid=[[141.99, 2.3823e-22],
                                               [130.0, 1.7407e-22],
                                               [121.84, 1.3031000000000001e-22],
                                               [115.0, 9.465e-23],
                                               [104.42, 6.630900000000001e-23],
                                               [91.2, 4.4507e-23],
                                               [91.17, 4.4507e-23],
                                               [80.0, 2.8438e-23]])),
        ExplicitContinuum(j=5, i=4, **convert_alphaGrid(alphaGrid=[[141.99, 2.3823e-22],
                                               [130.0, 1.7407e-22],
                                               [121.84, 1.3031000000000001e-22],
                                               [115.0, 9.465e-23],
                                               [104.42, 6.630900000000001e-23],
                                               [91.2, 4.4507e-23],
                                               [91.17, 4.4507e-23],
                                               [80.0, 2.8438e-23]]))
    ],
    collisions=[
        Omega(j=0, i=1, temperature=tempGrid, rates=[5.6, 5.6]),
        Omega(j=0, i=2, temperature=tempGrid, rates=[8.41, 8.41]),
        Omega(j=0, i=3, temperature=tempGrid, rates=[4.79, 4.79]),
        Omega(j=0, i=4, temperature=tempGrid, rates=[9.57, 9.57]),
        Omega(j=1, i=2, temperature=tempGrid, rates=[2.1e1, 2.1e1]),
        Omega(j=1, i=3, temperature=tempGrid, rates=[2.15e1, 2.15e1]),
        Omega(j=1, i=4, temperature=tempGrid, rates=[9.6, 9.6]),
        Omega(j=2, i=3, temperature=tempGrid, rates=[3.41, 3.41]),
        Omega(j=2, i=4, temperature=tempGrid, rates=[4.23e1, 4.23e1]),
        Omega(j=3, i=4, temperature=tempGrid, rates=[1e1, 1e1]),
        CI(j=5, i=0, temperature=tempGrid, rates=[1.45e-16, 1.45e-16]),
        CI(j=5, i=1, temperature=tempGrid, rates=[1.88e-16, 1.88e-16]),
        CI(j=5, i=2, temperature=tempGrid, rates=[1.88e-16, 1.88e-16]),
        CI(j=5, i=3, temperature=tempGrid, rates=[2.68e-16, 2.68e-16]),
        CI(j=5, i=4, temperature=tempGrid, rates=[2.68e-16, 2.68e-16]),
        CH(j=2, i=1, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 3.00e+05, 5.00e+05, 1.00e+06, 2.00e+06, 4.00e+06, 8.00e+06, 1.60e+07, 5.00e+07], rates=[1.31e-15, 1.48e-15, 1.83e-15, 2.29e-15, 2.86e-15, 3.59e-15, 4.51e-15, 5.55e-15, 5.55e-15, 5.55e-15, 5.55e-15, 5.55e-15, 5.55e-15, 5.55e-15, 5.01e-15, 5.01e-15]),
        CH(j=4, i=3, temperature=[1000.0, 3000.0, 6000.0, 12000.0, 24000.0, 48000.0, 96000.0, 192000.0, 3.00e+05, 5.00e+05, 1.00e+06, 2.00e+06, 4.00e+06, 8.00e+06, 1.60e+07, 5.00e+07], rates=[1.11e-15, 1.2e-15, 1.43e-15, 1.75e-15, 2.17e-15, 2.71e-15, 3.39e-15, 4.21e-15, 4.21e-15, 4.21e-15, 4.21e-15, 4.21e-15, 4.21e-15, 4.21e-15, 4.21e-15, 4.21e-15])
    ])

    Ca = CaII_radyn_nasa()
    # for l in Ca.lines:
    #     l.NlambdaGen *= 2

    # reconfigure_atom(Ca)
    # for r in radTrans:
        # if r['atomName'] == Ca.name:
            # if r['transType'] == 'Line':
                # Ca.lines[r['lwIdx']].wavelength = r['wlGrid'].value
                # Ca.lines[r['lwIdx']].preserveWavelength = True
    return Ca

# def He_9():
#     He = He_9_atom()
#     for l in He.lines:
#         l.NlambdaGen //= 2

#     reconfigure_atom(He)
#     # for r in radTrans:
#     #     if r['atomName'] == He.name:
#     #         if r['transType'] == 'Line':
#     #             He.lines[r['lwIdx']].wavelength = r['wlGrid'].value
#     #             He.lines[r['lwIdx']].preserveWavelength = True
#     return He

import numpy as np
from dataclasses import dataclass
from lightweaver.collisional_rates import CollisionalRates

@dataclass
class FangH:
    C1c: np.ndarray
    C12: np.ndarray
    C13: np.ndarray
    C14: np.ndarray


def fang_ele_rates_H(neutralH, ne1, bheat1):
    clog = 24.68
    clog1 = 8.13
    gam = ne1 * clog + neutralH * clog1
    coeff = np.maximum(bheat1, 0.0) * clog1 / gam

    C1c = 1.73e10 * coeff
    C12 = 2.94e10 * coeff
    C13 = 5.35e9 * coeff
    C14 = 1.91e9 * coeff

    return FangH(C1c, C12, C13, C14)

# NOTE(cmo): This is a little bit of a hacky way to deal with these rates, but it should work for now. We just need to ensure that hPops and bHeat are added to the atmos object --  I guess, in some ways this is the advantage of python and the whole point in Lightweaver
@dataclass
class FangHRates(CollisionalRates):
    def setup(self, atom):
        self.atom = atom

    def __repr__(self):
        return 'FangHRates(0,0)'

    def compute_rates(self, atmos, eqPops, Cmat):
        neutralH = np.sum(atmos.hPops[:-1, :], axis=0) / 1e6 # in cm-3
        # bHeat is left in default units
        fangRates = fang_ele_rates_H(neutralH, atmos.ne / 1e6, atmos.bHeat)
        Cmat[-1, 0, :] += fangRates.C1c
        Cmat[1, 0, :] += fangRates.C12
        Cmat[2, 0, :] += fangRates.C13
        Cmat[3, 0, :] += fangRates.C14

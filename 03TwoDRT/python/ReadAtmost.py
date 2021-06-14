import numpy as np
from dataclasses import dataclass

@dataclass
class Atmost:
    grav: float
    tau2: float
    vturb: np.ndarray

    time: np.ndarray
    dt: np.ndarray
    z1: np.ndarray
    d1: np.ndarray
    ne1: np.ndarray
    tg1: np.ndarray
    vz1: np.ndarray
    nh1: np.ndarray
    bheat1: np.ndarray

    cgs: bool = True

    def to_SI(self):
        if not self.cgs:
            return

        self.vturb /= 1e2
        self.z1 /= 1e2
        self.d1 *= 1e3
        self.ne1 *= 1e6
        self.vz1 /= 1e2
        self.nh1 *= 1e6

        # NOTE(cmo): we don't change the units on bheat1, since it's only used
        # for the Fang rates, which are entirely described with cgs.

        self.cgs = False


def read_atmost(filename='atmost.dat') -> Atmost:
    with open(filename, 'rb') as f:
        # Record: itype 4, isize 4, cname 8 : 16
        _ = np.fromfile(f, np.int32, 1)
        itype = np.fromfile(f, np.int32, 1)
        isize = np.fromfile(f, np.int32, 1)
        cname = np.fromfile(f, 'c', 8)
        _ = np.fromfile(f, np.int32, 1)

        # Record: ntime 4, ndep 4 : 8
        _ = np.fromfile(f, np.int32, 1)
        ntime = np.fromfile(f, np.int32, 1)
        ndep = np.fromfile(f, np.int32, 1)
        _ = np.fromfile(f, np.int32, 1)

        # Record: itype 4, isize 4, cname 8 : 16
        _ = np.fromfile(f, np.int32, 1)
        itype = np.fromfile(f, np.int32, 1)
        isize = np.fromfile(f, np.int32, 1)
        cname = np.fromfile(f, 'c', 8)
        _ = np.fromfile(f, np.int32, 1)

        # Record: grav 8, tau(2) 8, vturb 8 x ndep(300) : 2416
        _ = np.fromfile(f, np.int32, 1)
        grav = np.fromfile(f, np.float64, 1)
        tau2 = np.fromfile(f, np.float64, 1)
        vturb = np.fromfile(f, np.float64, ndep[0])
        _ = np.fromfile(f, np.int32, 1)
        if grav[0] == 0.0:
            grav[0] = 10**4.44

        times = []
        dtns = []
        z1t = []
        d1t = []
        ne1t = []
        tg1t = []
        vz1t = []
        nh1t = []
        bheat1t = []
        while True:
            # Record: itype 4, isize 4, cname 8 : 16
            _ = np.fromfile(f, np.int32, 1)
            itype = np.fromfile(f, np.int32, 1)
            isize = np.fromfile(f, np.int32, 1)
            cname = np.fromfile(f, 'c', 8)
            _ = np.fromfile(f, np.int32, 1)

            # Record: timep 8, dtnp 8, z1 8 * ndep(300),
            # d1 8 * ndep(300), ne1 8 * ndep(300),
            # tg1 8 * ndep(300), vz1 8 * ndep(300),
            # nh1 8 * 6 * ndep(300): 26416
            # bheat1 8 * ndep(300): 26416 + 2400
            recordSize = np.fromfile(f, np.int32, 1)
            if (recordSize - 16) / (8 * ndep[0]) == 11:
                bheat = False
            else:
                bheat = True
            times.append(np.fromfile(f, np.float64, 1))
            if times[-1].shape != (1,):
                times.pop()
                break
            dtns.append(np.fromfile(f, np.float64, 1))
            z1t.append(np.fromfile(f, np.float64, ndep[0]))
            d1t.append(np.fromfile(f, np.float64, ndep[0]))
            ne1t.append(np.fromfile(f, np.float64, ndep[0]))
            tg1t.append(np.fromfile(f, np.float64, ndep[0]))
            vz1t.append(np.fromfile(f, np.float64, ndep[0]))
            nh1t.append(np.fromfile(f, np.float64, ndep[0] * 6).reshape(6, ndep[0]))
            if bheat:
                bheat1t.append(np.fromfile(f, np.float64, ndep[0]))
            _ = np.fromfile(f, np.int32, 1)

    times = np.array(times).squeeze()
    dtns = np.array(dtns).squeeze()
    z1t = np.array(z1t).squeeze()
    d1t = np.array(d1t).squeeze()
    ne1t = np.array(ne1t).squeeze()
    tg1t = np.array(tg1t).squeeze()
    vz1t = np.array(vz1t).squeeze()
    nh1t = np.array(nh1t).squeeze()
    bheat1t = np.array(bheat1t).squeeze()

    return Atmost(grav.item(), tau2.item(), vturb, times, dtns, z1t, d1t, ne1t, tg1t, vz1t, nh1t, bheat1t)

def read_flarix(filename, filenameHPops, Ntime, Ndepth) -> Atmost:
    z1t = np.zeros((Ntime, Ndepth))
    tg1t = np.zeros((Ntime, Ndepth))
    ne1t = np.zeros((Ntime, Ndepth))
    d1t = np.zeros((Ntime, Ndepth))
    n1t = np.zeros((Ntime, Ndepth))
    vz1t = np.zeros((Ntime, Ndepth))
    nh1t = np.zeros((Ntime, Ndepth, 6))
    with open(filename, 'rb') as f:

        for t in range(Ntime):
            for k in range(Ndepth):
                _ = np.fromfile(f, np.int32, 1)
                z1t[t, k] = np.fromfile(f, np.float64, 1)
                tg1t[t, k] = np.fromfile(f, np.float64, 1)
                ne1t[t, k] = np.fromfile(f, np.float64, 1)
                n1t[t,k] = np.fromfile(f, np.float64, 1)
                d1t[t, k] = np.fromfile(f, np.float64, 1)
                _ = np.fromfile(f, np.float64, 1)
                vz1t[t, k] = np.fromfile(f, np.float64, 1)
                _ = np.fromfile(f, np.float64, 1)
                _ = np.fromfile(f, np.int32, 1)

    with open(filenameHPops, 'rb') as f:
        for t in range(Ntime):
            _ = np.fromfile(f, np.int32, 1)
            nh1t[t].reshape(-1)[...] = np.fromfile(f, np.float64, 6 * Ndepth)
            _ = np.fromfile(f, np.int32, 1)

    return Atmost(0, 0, vturb=2e5 * np.ones(Ndepth), time=np.arange(Ntime, dtype=np.float64) * 0.1,
                  dt=np.ones(Ntime) * 0.1, z1=-z1t, d1=d1t, ne1=ne1t, tg1=tg1t, vz1=-vz1t, nh1=nh1t, bheat1=np.array(()))





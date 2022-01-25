from visualizations.iVisualization import VisualizationInterface
from scipy.spatial import Voronoi
from skimage.draw import polygon
from controls.controllers import SkyMetaphorController
import holoviews as hv
import numpy as np
import panel as pn
import scipy.ndimage


class SkyMetaphor(VisualizationInterface):

    def __init__(self, main):
        self._main = main
        self._controls = SkyMetaphorController(self._calculate, name='SkyMetaphor visualization')

    def _activate_controllers(self, ):
        reference = pn.pane.Str("<ul><li><b>Title</b> Text</li></ul>")
        self._main._controls.append(pn.Column(self._controls, reference))
        self._calculate()

    def _deactivate_controllers(self, ):
        self._main._pipe_paths.send([])

    def _calculate(self, ):
        # density per unit
        dpu = 20
        height = self._main._m * dpu
        width = self._main._n * dpu
        k = 4
        lam = 0.25
        scale_matrix = np.array([dpu, dpu])
        stars = np.zeros((width, height))
        for vector in self._main._idata:
            dists = np.sqrt(np.sum(np.power(self._main._weights - vector, 2), axis=1))
            idxs = np.argsort(dists)
            idxs2 = np.array(list(map(lambda a: (int(a % self._main._n), int(a / self._main._n)), idxs)))
            best_dists = dists[idxs[0:k]]
            ui = idxs2[1:k]
            u1 = idxs2[0]
            position = np.zeros(2, dtype="float64")
            use4points = 1

            f = best_dists[0] / best_dists[1:]
            f = np.array(f).astype("float64")
            # use 4 points
            if (ui[0][0] == ui[1][0] and ui[1][0] == ui[2][0]
                    or ui[0][1] == ui[1][1] and ui[1][1] == ui[2][1]):
                use4points = 0
            for i in range(k - 1 - use4points):
                if (ui[i][0] - u1[0]) != 0:
                    position[0] += f[i] * 1.0 / (ui[i][0] - u1[0])
                if (ui[i][1] - u1[1]) != 0:
                    position[1] += f[i] * 1.0 / (ui[i][1] - u1[1])
            position *= lam
            position *= dpu

            position[0] = position[0] + u1[0] * dpu
            position[1] = position[1] + u1[1] * dpu
            x = int(position[0] + dpu/2.0)
            y = int(position[1] + dpu/2.0)
            stars[x, y] = 1

        res = self.sdh(4, 2)
        res = scipy.ndimage.zoom(res, dpu, order=2)
        res = np.array(res)
        res = res / np.max(res)
        #res[res > 1] = 1
        stars = stars.transpose()
        res *= 0.6
        res = 1 - res
        res[stars == 1] = 0
        self._main._display(res)

    def sdh(self, smooth_factor, sdh_type):
        import heapq

        sdh_m = np.zeros(self._main._m * self._main._n)

        cs = 0
        for i in range(smooth_factor): cs += smooth_factor - i

        for vector in self._main._idata:
            dist = np.sqrt(np.sum(np.power(self._main._weights - vector, 2), axis=1))
            c = heapq.nsmallest(smooth_factor, range(len(dist)), key=dist.__getitem__)
            if (sdh_type == 0):
                for j in range(smooth_factor):  sdh_m[c[j]] += (smooth_factor - j) / cs  # normalized
            if (sdh_type == 1):
                for j in range(smooth_factor): sdh_m[c[j]] += 1.0 / dist[c[j]]  # based on distance
            if (sdh_type == 2):
                dmin = min(dist[c])
                dmax = max(dist[c])
                for j in range(smooth_factor): sdh_m[c[j]] += 1.0 - (dist[c[j]] - dmin) / (dmax - dmin)

        plot = sdh_m.reshape(self._main._m, self._main._n)
        return plot
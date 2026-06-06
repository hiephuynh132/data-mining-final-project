from .kmeans_algo import KMeansAlgo
from .diskmeans   import DisKmeans
from .fdkm        import FDKM
from .ifdkm       import IFDKM
from .ifdfd       import IFDFD

ALGO_REGISTRY = {
    "kmeans":    KMeansAlgo,
    "diskmeans": DisKmeans,
    "fdkm":      FDKM,
    "ifdkm":     IFDKM,
    "ifdfd":     IFDFD,
}

KERNEL_ALGOS = {"diskmeans", "fdkm", "ifdkm", "ifdfd"}  # share random init

__all__ = ["KMeansAlgo", "DisKmeans", "FDKM", "IFDKM", "IFDFD",
           "ALGO_REGISTRY", "KERNEL_ALGOS"]

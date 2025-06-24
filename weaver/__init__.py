__version__ = "0.1.0"

try:
    from .models import Model
    from .dataset import VerificationDataset, ClusteringDataset
except ImportError:
    pass

__all__ = [
    "Model",
    "VerificationDataset", 
    "ClusteringDataset",
]
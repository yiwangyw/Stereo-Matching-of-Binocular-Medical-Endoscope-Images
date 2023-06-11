from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .SCARED_dataset import ScaredDatset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "scared": ScaredDatset
}

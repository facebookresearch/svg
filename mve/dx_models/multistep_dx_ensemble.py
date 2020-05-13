from torch.nn import ModuleList

from model_zoo.ensemble import BaseEnsemble
from mve.dx_models import MultistepDx


class MultistepDxEnsemble(BaseEnsemble):
    """ Ensemble of fully-connected neural net regression models
    """
    def __init__(self, input_dim, target_dim, num_components,
                 num_elites, model_class, model_kwargs, mode='prob', obs_dim=None):
        """
        Args:
            input_dim (int)
            target_dim (int)
            num_components (int)
            num_elites (int)
            submodule_params (dict): kwargs to pass to FC constructor)
        """
        super().__init__(input_dim, target_dim, num_components, num_elites)
        components = [
            MultistepDx(
                input_dim,
                target_dim,
                model_class,
                model_kwargs,
                mode,
                obs_dim
            ) for _ in range(num_components)
        ]
        self.components = ModuleList(components)
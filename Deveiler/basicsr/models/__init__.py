import importlib
from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# Import only DeVeiler training models. This keeps the registry stable after
# archiving unrelated experiment modules.
model_filenames = [
    'image_restoration_model',
    'image_restoration_wlpe_model',
    'image_restoration_wlpe_reblur_model',
]
_model_modules = [importlib.import_module(f'basicsr.models.{file_name}') for file_name in model_filenames]


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model

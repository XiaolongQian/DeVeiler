import importlib
from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# Import only DeVeiler training architectures. Required helper blocks are
# inlined in deveiler_arch for the minimal reproducible release.
arch_filenames = ['deveiler_arch']
_arch_modules = [importlib.import_module(f'basicsr.archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net

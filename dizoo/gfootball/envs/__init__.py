import warnings

try:
    from .gfootball_academy_env_ce import GfootballAcademyEnv
except ImportError:
    warnings.warn("not found gfootball env, please install it")
    GfootballEnv = None

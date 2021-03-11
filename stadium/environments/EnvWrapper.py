from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stadium.settings import GlobalConfig as settings
import stadium.environments
import gym
import numpy as np
import cv2
import time
# import procgen
import gym_sokoban
from stadium.settings import GlobalConfig as config

def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor
# disable_view_window()

def decorator(env_name, **kwargs):
    """
    Decorator for gym environments
    """
    def _init():
        env = gym.make(env_name)
        return env
    return _init

def pybdecorator(env_name, **kwargs):
    """
    Decorator for gym environments
    """
    def _init():
        env = gym.make(env_name)
        env.render(mode='human')
        env.reset()
        return env
    return _init

def customdec(env_name, config=None, **kwargs):
    def _init():
        env_obj = getattr(stadium.environments.custom, env_name)
        env = env_obj(**kwargs)
        return env
    return _init

def create_env(name, multi, **kwargs):

    try:
        env = gym.make(name)
        id = env.spec.id
        if 'procgen' in id:
            if multi:
                return MultiProcgenWrapper(name, **kwargs)
            else:
                return ProcgenWrapper(name, **kwargs)
        elif hasattr(env.unwrapped, 'ale'):
            if multi:
                return MultiAtariWrapper(name, **kwargs)
            else:
                return AtariWrapper(name, **kwargs)
        elif 'okoban' in id:
            if multi:
                return MultiSokobanWrapper(name, **kwargs)
            else:
                return SokobanWrapper(name, **kwargs)
        else:
            if multi:
                return MultiGymClassicWrapper(name, **kwargs)
            else:
                return GymClassicWrapper(name, **kwargs)
    except Exception as e:
        if multi:
            return MultiCustomWrapper(name, **kwargs)
        else:
            return CustomWrapper(name, **kwargs)


def tile_image(self, state):
    state = state.squeeze()
    dims = len(state.shape)
    if state.shape[0] == 1:
        state = np.flip(np.rot90(state), axis=0)
        return state.squeeze()
    if dims == 3:
        n, h, w = state.shape
    elif dims == 4:
        n, h, w, c = state.shape

    for i in range(1, 8):
        if i**2 <= n <= (i+1)**2:
            x = i + 1
            y = round(n/x)
            break
    if dims == 3:
        tiled = np.zeros([x*w, y*h])
    else:
        tiled = np.zeros([x*w, y*h, c])

    for i in range(x):
        for j in range(y):
            if i*x+j >= n:
                break
            if dims > 3:
                t = state[i*x+j, :, :, :]
                t = np.swapaxes(t, 0, 1)
                tiled[w*i:w*(i+1), h*j:h*(j+1), :] = t
            else:
                tiled[w*i:w*(i+1), h*j:h*(j+1)] = state[i*x+j, :, :]
    return tiled


class EnvWrapper(DummyVecEnv):
    def __init__(self, name, config=None, n_workers=1, dec_fn=decorator, **kwargs):
        self.name = name
        self.config = config
        self.workers = n_workers
        self.env_type = ''
        vectorized = [dec_fn(name, config=config, **kwargs) for x in range(n_workers)]
        super(EnvWrapper, self).__init__(vectorized)


class MultiEnvWrapper(DummyVecEnv):
    def __init__(self, name, config=None, n_workers=1, dec_fn=decorator, **kwargs):
        self.name = name
        self.config = config
        self.workers = n_workers
        self.env_type = ''
        vectorized = [dec_fn(name, config=config, **kwargs) for x in range(n_workers)]
        super(MultiEnvWrapper, self).__init__(vectorized)


class GymClassicWrapper(EnvWrapper):
    def __init__(self, name, n_workers=1):
        super(GymClassicWrapper, self).__init__(name, n_workers=n_workers)
        self.reset()
        self.render()
        self.img = np.zeros([3, 3, 3])
        
    def get_image(self, image=None):
        self.render()

        return self.img



class AtariWrapper(EnvWrapper):
    def __init__(self, name, n_workers=1):
        super(AtariWrapper, self).__init__(name, n_workers=n_workers)

    def get_image(self, image=None):
        if not image:
            image = self.render('rgb_array')
        image = np.flip(np.rot90(image), axis=0)
        return image.squeeze()


class ProcgenWrapper(EnvWrapper):
    def __init__(self, name, n_workers=1):
        super(ProcgenWrapper, self).__init__(name, n_workers=n_workers)
        self.env_type = 'procgen'

    def get_image(self, image=None):
        if not image:
            image = self.reset()
        if image.shape[0] != 1:
            image = tile_image(image)
        else:
            image = image.squeeze()
        image = np.flip(np.rot90(image), axis=0)
        return image

class CustomWrapper(EnvWrapper):
    def __init__(self, name, config=None, n_workers=1, **kwargs):
        super(CustomWrapper, self).__init__(name, config=config, n_workers=n_workers, dec_fn=customdec, **kwargs)

    def get_image(self, image=None):
        if not image:
            image = np.stack(self.get_attr('grid'), axis=0)
        return image.squeeze()


class MultiGymClassicWrapper(MultiEnvWrapper):
    def __init__(self, name, n_workers=1):
        super(MultiGymClassicWrapper, self).__init__(name, n_workers=n_workers)
        self.reset()
        self.img = np.zeros([3, 4, 4])
        # self.render(mode='rgb_array')
        
    def get_image(self, image=None):
        if not image:
            self.render()
        return self.img

class MultiAtariWrapper(MultiEnvWrapper):
    def __init__(self, name, n_workers=4):
        super(MultiAtariWrapper, self).__init__(name, n_workers=n_workers)

    def get_image(self, image=None):
        if not image:
            image = self.render('rgb_array')
        image = np.flip(np.rot90(image), axis=0)
        return image.squeeze()


class MultiProcgenWrapper(MultiEnvWrapper):
    def __init__(self, name, n_workers=4):
        super(MultiProcgenWrapper, self).__init__(name, n_workers=n_workers)
        self.env_type = 'procgen'

    def get_image(self, image=None):
        if not image:
            image = self.reset()
        if image.shape[0] != 1:
            image = tile_image(image)
        else:
            image = image.squeeze()
        image = np.flip(np.rot90(image), axis=0)
        return image

class MultiCustomWrapper(MultiEnvWrapper):
    def __init__(self, name, config=None, n_workers=4, **kwargs):
        super(MultiCustomWrapper, self).__init__(name, config=config, n_workers=n_workers, dec_fn=customdec, **kwargs)

    def get_image(self, image=None):
        if not image:
            image = np.stack(self.get_attr('grid'), axis=0)
        return image

class VizdoomWrapper(EnvWrapper):
    def __init__(self, name, n_workers=2):
        super(VizdoomWrapper, self).__init__(name, n_workers=n_workers)

    def get_image(self, image=None):
        if not image:
            image = self.reset()
        print(image.shape)
        image = tile_image(image)
        return image

class MultiSokobanWrapper(MultiEnvWrapper):
    def __init__(self, name, n_workers=4):
        super(MultiSokobanWrapper, self).__init__(name, n_workers=n_workers)

    def get_image(self, image=None):
        if not image:
            image = self.render(mode='rgb_array')
        print(image.shape)
        image = image.squeeze()
        print(image.shape)
        image = np.flip(np.rot90(image), axis=0)
        return image


class SokobanWrapper(EnvWrapper):
    def __init__(self, name, n_workers=1):
        super(SokobanWrapper, self).__init__(name, n_workers=n_workers)

    def get_image(self, image=None):
        if not image:
            image = self.render(mode='rgb_array')
        print(image.shape)
        image = image.squeeze()
        print(image.shape)
        image = np.flip(np.rot90(image), axis=0)
        return image
## Deepdrive

def deepdrivedec(map_name, scenario, **kwargs):
    def _init():
        env = sim.start(map=map_name, scenario_index=scenario)
        return env
    return _init
    
class DeepDriveWrapper(DummyVecEnv):
    def __init__(self, map='kevindale_bare', scenario_index=1):
        self.map = map
        self.scenario = scenario_index
        vec = [deepdrivedec(self.map, self.scenario)]
        super(DeepDriveWrapper, self).__init__(vec)

    def get_image(self):
        return None
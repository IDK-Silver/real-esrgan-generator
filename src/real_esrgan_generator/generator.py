from enum import Enum

import glob
import os
from typing import Union

import cv2
import numpy
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class RealESRGANModelType(Enum):
    real_esr_gan_x4_plus = 'RealESRGAN_x4plus'
    real_esr_net =  'RealESRNet_x4plus'
    real_esr_gan_x4_plus_anime =  'RealESRGAN_x4plus_anime_6B'
    real_esr_gan_x2_plus =  'RealESRGAN_x2plus'
    real_esr_anime_video_v3 =  'realesr-animevideov3'
    real_esr_general_x4_v3 = 'realesr-general-x4v3'

class RealESRGANGeneratorConfig:
    def __init__(self):
        self.denoise_strength = None
        self.tile = 0
        self.tile_pad = 0
        self.pre_pad = 0
        self.fp32 = False
        self.gpu_id = None
        self.outscale = 4
        self.face_enhance = False

class RealESRGANGenerator:
    def __init__(self, config: RealESRGANGeneratorConfig):
        self.model = None
        self.upsampler = None
        self.face_enhancer = None
        self.netscale = None
        self.denoise_strength = config.denoise_strength
        self.tile = config.tile
        self.tile_pad = config.tile_pad
        self.pre_pad = config.pre_pad
        self.half = config.pre_pad
        self.fp32 = config.fp32
        self.gpu_id = None
        self.outscale = config.outscale
        self.face_enhance = config.outscale

    def load_model(self, type: RealESRGANModelType, model_path: str = None):
        file_url = []

        if type.value == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif type.value == 'RealESRNet_x4plus':  # x4 RRDBNet model
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif type.value == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            self.netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif type.value == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self.netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif type.value == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
            self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            self.netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif type.value == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            self.netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]
        if model_path is None or len(model_path) == 0:
            model_path = os.path.join('weights', type.value + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
        #
        # # determine model paths
        # if model_path is None and auto_download is True:
        #     model_path = os.path.join('weights', type.value + '.pth')
        #     if not os.path.isfile(model_path):
        #         root_dir = os.path.dirname(os.path.abspath(__file__))
        #         for url in file_url:
        #             # model_path will be updated
        #             model_path = load_file_from_url(
        #                 url=url, model_dir=os.path.join(root_dir, 'weights'), progress=True, file_name=None)

        # use dni to control the denoise strength
        dni_weight = None
        if type.value == 'realesr-general-x4v3' and self.denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [self.denoise_strength, 1 - self.denoise_strength]

        # restorer
        self.upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=self.model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=not self.fp32,
            gpu_id=self.gpu_id)

        if self.face_enhance:  # Use GFPGAN for face enhancement
            from gfpgan import GFPGANer
            self.face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=self.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler)


    def generate(self, image: Union[str, numpy.array]):

        # if user given a path then load image from path
        if isinstance(image, str):
            image = cv2.imread(image)
        assert image is not None, 'Image Not Found '

        # BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if self.face_enhance:
                _, _, image = self.face_enhancer.enhance(image, has_aligned=False, only_center_face=False,
                                                     paste_back=True)
            else:
                image, _ = self.upsampler.enhance(image, outscale=self.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')

        return image


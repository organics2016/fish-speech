from unittest import TestCase
from fish_speech.lib import Pipeline
import sounddevice as sd
import time
import numpy as np

"""
conda install "ffmpeg"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .[cu128]
pip install torchcodec
pip install sounddevice
"""


class TestPipeline(TestCase):

    def setUp(self):
        self.tts_model = Pipeline(
            llama_path="checkpoints/openaudio-s1-mini",
            vqgan_path="checkpoints/openaudio-s1-mini/codec.pth",
            device='cuda',
            compile=False
        )
        self.ref_audio = self.tts_model.make_reference("test/t1.wav",
                                                       "你今后将成为爱丽丝，这并不是因为叫你爱丽丝才变成爱丽丝的，而是，你根据你自己的意愿成为爱丽丝的。")

    def test_generate(self):

        for i in range(2):
            t0 = time.time()
            audio = self.tts_model.generate('OpenAudio 支持多种安装方式，请选择最适合您开发环境的方法。', self.ref_audio,
                                            streaming=False)
            print(f'speak generator time:{(time.time() - t0):.02f}')

            # 声道验证
            if audio.ndim <= 0 or audio.ndim > 2:
                raise ValueError('Audio channel verification failed. Only single or dual channels are supported.')

            with sd.OutputStream(samplerate=self.tts_model.sample_rate,
                                 blocksize=4096,
                                 device=3,
                                 channels=audio.ndim,  # if data.shape[1] != channels:
                                 dtype=audio.dtype) as stream:
                stream.write(audio)

    def test_streaming(self):
        for i in range(1):
            t0 = time.time()
            generator = self.tts_model.generate('Ori是我的一个好朋友，我第一次听到Ori这个名字，你觉的这个名字怎么样？',
                                                self.ref_audio, seed=111, chunk_length=20,
                                                num_samples=10,
                                                streaming=True)
            print(f'speak generator time:{(time.time() - t0):.02f}')

            with sd.OutputStream(samplerate=self.tts_model.sample_rate,
                                 blocksize=4096,
                                 device=3,
                                 channels=1,  # if data.shape[1] != channels:
                                 dtype=np.float32) as stream:

                for part in generator:
                    print("part:ffffff")
                    stream.write(part)

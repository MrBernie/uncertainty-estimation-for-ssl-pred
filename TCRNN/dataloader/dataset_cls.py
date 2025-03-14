import pickle
import os

import torch
from torch.utils.data import Dataset

import numpy as np
from numpy import ndarray
from scipy.signal import stft

import soundfile as sf
from dataloader.Dataset import AcousticScene
from utils.simu import Segmenting_SRPDNN
from loguru import logger

import webrtcvad
import librosa

class TSSLDataSet(Dataset):
    def __init__(
        self,
        data_dir,
        num_data,
        return_acoustic_scene=False,
        stage="fit",
        pred_result_dir: str = None,
                 ):
        super().__init__()

        self.stage = stage
        self.vad = webrtcvad.Vad() # init for VAD
        self.data_paths = []
        data_names = os.listdir(data_dir)
        for fname in data_names:
            front, ext = os.path.splitext(fname)
            if ext == ".wav":
                self.data_paths.append((os.path.join(data_dir, fname)))
        self.num_data = len(self.data_paths) if num_data is None else num_data
        self.gt_segmentation = Segmenting_SRPDNN(
            K=3328, # int(win_len512*win_shift_ratio0.5*(seg_fra_ratio12+1))
            step=3072, # int(win_len*win_shift_ratio*(seg_fra_ratio))
            window=None
        )
        self.acoustic_scene = AcousticScene(
            room_sz = [],
            T60 = [],
            beta = [],
            noise_signal = [],
            SNR = [],
            source_signal = [],
            fs = [],
            array_setup = [],
            mic_pos = [],
            timestamps = [],
            traj_pts = [],
            trajectory = [],
            t = [],
            DOA = [],
            c = [],
        )
        self.return_acoustic_scene = return_acoustic_scene
        self.pred_result_dir = pred_result_dir
    
    def _get_audio_features(self,
                            audio_file: str,
                            ) -> ndarray:
        """Computes spectrogram audio features for a given chunk from an audio file.

        Args:
            audio_file (str): Path to audio file in *.wav format.
            start_time (float): Chunk start time in seconds.
            end_time (float): Chunk end time in seconds.

        Returns:
            ndarray: Spectrogram audio features.
        """
        file_info = sf.info(audio_file)
        audio_data, samp_freq = sf.read(audio_file)
        # print(audio_data.shape)
        print(audio_file)
        
        # down sample the audio data to 16kHz
        if samp_freq != 16000:
            audio_data, samp_freq = librosa.load(audio_file, sr=16000, mono=False)
            audio_data = audio_data.T
            samp_freq = 16000
            # print(audio_data.shape)
        # Compute multi-channel STFT and remove first coefficient and last frame

        # If the stage is prediction, apply VAD before using STFT
        if self.stage == "pred":
            # VAD process for the prediction data
            # print(audio_data.shape)
            audio_data_clean, vad_out = self._cleanSilences(audio_data, samp_freq, aggressiveness=3, return_vad=True)
            # print(np.count_nonzero(audio_data_clean[:,0]), len(audio_data_clean))

            if np.count_nonzero(audio_data_clean[:,0]) < len(audio_data_clean) * 0.66:
                audio_data_clean, vad_out = self._cleanSilences(audio_data, samp_freq, aggressiveness=2, return_vad=True)
                # print(np.count_nonzero(audio_data_clean[:,0]), len(audio_data_clean))

            if np.count_nonzero(audio_data_clean[:,0]) < len(audio_data_clean) * 0.66:
                audio_data_clean, vad_out = self._cleanSilences(audio_data, samp_freq, aggressiveness=1, return_vad=True)
                # print(np.count_nonzero(audio_data_clean[:,0]), len(audio_data_clean))

            if np.sum(vad_out) == 0:
                logger.warning(f"No speech detected in {audio_file}")
                return None  # return None for no speech detected
            
            audio_data = audio_data_clean
            pred_result_dir = self.pred_result_dir
            vad_out_dir = os.path.join(pred_result_dir, "vad_out")
            if not os.path.exists(vad_out_dir):
                os.makedirs(vad_out_dir)

            # save the vad result to files
            audio_filename = os.path.basename(audio_file)
            audio_filename = os.path.splitext(audio_filename)[0]
            try:
                vad_out_file = os.path.join(vad_out_dir, f'{audio_filename}-vad-out.txt')
                np.savetxt(vad_out_file, vad_out.astype(int), fmt='%d', delimiter=',')
            except Exception as e:
                print(f"Failed to save VAD output: {e}")
            
            
        # print(audio_data.shape)
        spectrogram = stft(audio_data,
                           fs=file_info.samplerate,
                           nperseg=512,
                           nfft=512,
                           padded=False,
                           axis=0)[-1] # [1025， 4， 101] 1025: the frequencies; 101: the times; 4: the channels
        spectrogram = spectrogram[1:, :, :-1] # [1024, 4, 100]
        spectrogram = spectrogram.transpose([1, 0, 2]) # [4, 100, 1024]
        spectrogram_real = np.real(spectrogram)
        spectrogram_img = np.imag(spectrogram)
        audio_features = np.concatenate((spectrogram_real, spectrogram_img),axis=0) # 4, 299, 256
        # print(audio_features.shape)
        return audio_features.astype(np.float32), file_info.samplerate
    
    def _gt_acoustic_scene(self,
                           acous_path,):

        file = open(acous_path,'rb')
        dataPickle = file.read()
        file.close()
        self.acoustic_scene.__dict__ = pickle.loads(dataPickle)
        return self.acoustic_scene

    def _cleanSilences(self, s, sample_rate, aggressiveness=3, return_vad=False):
        """VAD pre-processing of the prediction audio signal."""
        self.vad.set_mode(aggressiveness)

        vad_out = np.zeros_like(s)  # init VAD output
        vad_frame_len = int(10e-3 * sample_rate)  # 10ms every frame
        n_vad_frames = len(s) // vad_frame_len  # calculate the number of frames

        # VAD every frame
        for frame_idx in range(n_vad_frames):
            frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
            frame_bytes = (frame * 32767).astype('int16').tobytes()  # convert to bytes
            vad_out[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len] = self.vad.is_speech(frame_bytes, sample_rate)

        s_clean = s * vad_out  # apply the VAD mask
        return (s_clean, vad_out) if return_vad else s_clean

    def __len__(self):
        return self.num_data
    def __getitem__(self, idx):

        audio_path = self.data_paths[idx]
        audio_feat, sample_rate = self._get_audio_features(audio_path)
            
        if self.stage == "pred":
            # audio_path = self.data_paths[idx]
            # audio_feat = self._get_audio_features(audio_path)

            file_name = os.path.basename(audio_path)
            front, ext = os.path.splitext(file_name)
            # print(audio_feat.shape)
            return audio_feat, front
            
        audio_path = self.data_paths[idx]
        acous_path = audio_path.replace("wav", "npz")

        acous_scene = self._gt_acoustic_scene(acous_path)

        audio_feat_, acous_scene_ = self.gt_segmentation(
            audio_feat,
            acous_scene
        )

        vad_gt = acous_scene_.mic_vad_sources.mean(axis=1) # [24, 1]

        gts = {}
        gts["doa"] = acous_scene_.DOAw.astype(np.float32)
        gts["vad_sources"] = vad_gt.astype(np.float32)
        # logger.debug(f"vad sources shape: {vad_gt.shape}")

        print("DOA: ",gts["doa"])

        return audio_feat_, gts
if __name__ == "__main__":
    data_dir = "/home/data/DCASE2021-task3-dev/foa_dev"
    dataset = TSSLDataSet(data_dir, num_data=1)
    for i in range(len(dataset)):
        audio_feat, gts = dataset[i]
        print(audio_feat.shape)
        print(gts["doa"].shape)
        print(gts["vad_sources"].shape)
        break
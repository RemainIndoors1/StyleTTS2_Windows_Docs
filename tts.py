from pathlib import Path
import librosa
import scipy
import torch
import torchaudio
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import yaml

import models
import utils
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
import subprocess
import unicodedata

SINGLE_INFERENCE_MAX_LEN = 420

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class StyleTTS2:
    def __init__(self, model_checkpoint_path=None, config_path=None):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = None
        self.model_params = None
        self.model = self.load_model(model_path=model_checkpoint_path, config_path=config_path)

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )


    def load_model(self, model_path=None, config_path=None):
        """
        Loads model to prepare for inference. Loads checkpoints from provided paths or from local cache (or downloads
        default checkpoints to local cache if not present).
        :param model_path: Path to LibriTTS StyleTTS2 model checkpoint (TODO: LJSpeech model support)
        :param config_path: Path to LibriTTS StyleTTS2 model config JSON (TODO: LJSpeech model support)
        :return:
        """

        self.config = yaml.safe_load(open(config_path))

        # load pretrained ASR model
        ASR_config = self.config.get('ASR_config', False)
        print("ASR_config", ASR_config)
        if not ASR_config:
            exit()
        ASR_path = self.config.get('ASR_path', False)
        print("ASR_Path:", ASR_path)
        if not ASR_path:
            exit()

        text_aligner = models.load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = self.config.get('F0_path', False)
        print("F0 path:", F0_path)
        if not F0_path:
            exit()

        pitch_extractor = models.load_F0_models(F0_path)

        # load BERT model
        BERT_dir_path = self.config.get('PLBERT_dir', False)  # Directory at BERT_dir_path should contain PLBERT config.yml AND checkpoint
        print("BERT_dir_path:", BERT_dir_path)
        if not BERT_dir_path:
            exit()
        else:
            plbert = load_plbert(BERT_dir_path)

        self.model_params = utils.recursive_munch(self.config['model_params'])
        model = models.build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        _ = [model[key].eval() for key in model]
        _ = [model[key].to(self.device) for key in model]

        params_whole = torch.load(model_path, map_location='cpu')
        params = params_whole['net']

        for key in model:
            if key in params:
                print('%s loaded' % key)
                try:
                    model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [model[key].eval() for key in model]

        return model


    def compute_style(self, path):
        """
        Compute style vector, essentially an embedding that captures the characteristics
        of the target voice that is being cloned
        :param path: Path to target voice audio file
        :return: style vector
        """
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def phonemize_with_espeak(self, text):
        result = subprocess.run(
            ['espeak-ng', '-q', '--ipa=3', text],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        # Combine lines into one
        ipa_raw = result.stdout.strip().replace('\n', ' ').replace('\r', ' ')

        # Normalize and remove invisible characters (like ZERO WIDTH JOINER)
        ipa_clean = unicodedata.normalize('NFKD', ipa_raw)
        ipa_clean = ''.join(c for c in ipa_clean if not unicodedata.category(c).startswith('C'))

        # Collapse multiple spaces
        ipa_clean = ' '.join(ipa_clean.split())

        return ipa_clean

    def inference(self,
                  text: str,
                  target_voice_path=None,
                  output_wav_file=None,
                  output_sample_rate=24000,
                  alpha=0.3,
                  beta=0.7,
                  diffusion_steps=5,
                  embedding_scale=1,
                  speed=1.0,
                  ref_s=None):
        """
        Text-to-speech function
        :param text: Input text to turn into speech.
        :param target_voice_path: Path to audio file of target voice to clone.
        :param output_wav_file: Name of output audio file (if output WAV file is desired).
        :param output_sample_rate: Output sample rate (default 24000).
        :param alpha: Determines timbre of speech, higher means style is more suitable to text than to the target voice.
        :param beta: Determines prosody of speech, higher means style is more suitable to text than to the target voice.
        :param diffusion_steps: The more the steps, the more diverse the samples are, with the cost of speed.
        :param embedding_scale: Higher scale means style is more conditional to the input text and hence more emotional.
        :param ref_s: Pre-computed style vector to pass directly.
        :return: audio data as a Numpy array (will also create the WAV file if output_wav_file was set).
        """

        if ref_s is None:
            ref_s = self.compute_style(target_voice_path)  # target style vector

        text = text.strip()
        text = text.replace('"', '')
        phoneme_string = self.phonemize_with_espeak(text)

        textcleaner = TextCleaner()
        tokens = textcleaner(phoneme_string)

        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                  embedding=bert_dur,
                                  embedding_scale=embedding_scale,
                                  features=ref_s, # reference from the same speaker as the embedding
                                  num_steps=diffusion_steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            # duration prediction
            d = self.model.predictor.text_encoder(d_en,
                                                  s, input_lengths, text_mask)
            #print(f"d_en: {d_en} | s: {s} | input_lengths: {input_lengths} | text_mask: {text_mask}")
            x, _ = self.model.predictor.lstm(d)

            # duration head -> logits
            logits = self.model.predictor.duration_proj(x)  # e.g., [1, T_tokens, T_frames]
            probs = torch.sigmoid(logits)
            duration = probs.sum(dim=-1)  # [1, T_tokens] or [T_tokens]

            # make it [T_tokens]
            duration = duration.squeeze(0) if duration.dim() == 2 else duration

            # Apply speed AFTER sigmoid-sum (speed<1=faster, >1=slower)
            scaled = duration / max(float(speed), 1e-6)

            # clean & quantize
            scaled = torch.nan_to_num(scaled, nan=1.0, posinf=1e5, neginf=1.0)  # no NaNs/Infs
            pred_dur = torch.round(scaled).clamp(min=1)  # [T_tokens], dtype=float

            # Build alignment
            tokens_len = int(input_lengths.item() if torch.is_tensor(input_lengths) else input_lengths)
            total_frames = int(pred_dur.sum().item())
            pred_aln_trg = torch.zeros(tokens_len, total_frames, device=pred_dur.device)

            c_frame = 0
            for i in range(tokens_len):
                dur_i = int(pred_dur[i].item())  # <- .item(), not .data
                pred_aln_trg[i, c_frame:c_frame + dur_i] = 1
                c_frame += dur_i

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,
                                     F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        output = out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later
        if output_wav_file:
            scipy.io.wavfile.write(output_wav_file, rate=output_sample_rate, data=output)
        return output

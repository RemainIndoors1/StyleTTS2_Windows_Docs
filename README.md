# StyleTTS2 Training on (mostly) Windows

## Overview

If you're looking into Text-to-Speech solutions, and you've done any reading on the topic of [StyleTTS2](https://github.com/yl4579/StyleTTS2), you likely already know how amazing it can sound for something you can host yourself. I'm primarily a .Net developer, and generally prefer Windows and exhaustive documentation as opposed to breadcrumbs and half-answers.

Python is an extremely powerful tool as is Linux, but there seems to be an elitist mindset for both that makes it difficult to get started when you're only comfortable with Windows or haven't spent years trying to train TTS models. As such, I've spent a considerable amount of time figuring out how to train StyleTTS2 in Windows and run inference, so I figured I would document my findings here as they might be useful for others, or even for myself later when I forget what I did.

If you're planning on training a StyleTTS2 model from scratch, you've got some work ahead of you, but it can be a fun learning experience if you're brand new to training AI models and TTS in general. Note: This guide is going to document how I trained a model from scratch. There are definitely other ways to do it, but this has worked for me, so apologies for any unintentionally inaccurate statements.

## Getting Started

Requirements:
- You're going to need an Nvidia GPU with a lot of VRAM. I'm using a 3090 with 24GB VRAM, and it honestly still isn't enough.
- You'll probably also want to have a fairly powerful CPU.
- You'll need to install [Python](https://www.python.org/downloads/windows/) 3.7 or newer (I'm using 3.10.11)
- You'll need to install [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) to act as a phonemizer. (regular espeak should work too, but this method uses espeak-ng)
	- what's a phonemizer? basically, it takes text and converts it to base units that StyleTTS2 will know how to pronounce. For example, "tough" sounds like "tuff", "bough" sounds like "bow" and "bought" sounds like "bot". They're all spelled similarly, but TTS doesn't inherently know there's a difference.
- You'll need a considerable amount of clean voice data cut up into 2-10 second wav files with a 24kHz sample rate. (we'll cover how to convert wav files to 24kHz later)
- You'll also need text Transcripts for all of your wav files.
- You'll need to install the [CUDA toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive) to allow your python code to access the full power of your GPU.
- You might also need [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) to convert your wav files to the correct sample rate. This depends on what format they're currently in.

First, you need to clone the [StyleTTS2](https://github.com/yl4579/StyleTTS2) repository from GitHub. I'm not going to go over how to do this, but you can find a lot of info online for how to clone a github repository if you're not sure how.

Next, you'll want to add espeak-ng to the Path variable in your environment variables in Windows. This will give your python code access to the program to use as a phonemizer.

Also, If you need to convert your wav files to 24kHz, you'll want to add ffmpeg to the Path variable in your environment variables. This will make converting wav files much easier.

[espeak-ng](https://github.com/espeak-ng/espeak-ng/releases)
- download the msi file and install it. Windows will give warnings because it's not licensed with Microsoft, so proceed at your own risk, but you're already dealing with a lot of open source software, if you don't trust one piece of the process, this probably isn't for you.
- On Windows 11, this installs to C:\Program Files\eSpeak NG
- Add this folder as a new entry to the System "Path" variable in Environment Variables if it's not already there from installation.

[ffmpeg](https://www.gyan.dev/ffmpeg/builds/) 
- download the latest ffmpeg-git-full.7z zip file, open it and copy the contents to a new local folder on your PC (C:\ffmpeg should be easy enough)
- Also add this folder as a new entry to the System "Path" variable in Environment Variables.

You can verify both of those changes worked by opening a command prompt and typing `espeak-ng` then pressing enter, then typing `ffmpeg` and pressing enter. You might need to type `CTRL+C` to cancel one or both of those commands, but as long as it doesn't say something like `espeak-ng is not recognized as an internal or external command, operable program or batch file.` then it should be setup and working.

[CUDA toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- Download and install the version for your specific version of windows.
- Annoyingly enough, you're probably going to run into some more errors later about one or more missing cudnn files, so let's try to avoid that in advance. First you'll need to download a zip file from Nvidia's website and copy some files into the bin folder of your NVIDIA GPU Computing Toolkit installation.
	- Here's the installation instructions for installing the cuDNN Backend on Windows: https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html
	- you'll need to download the zip file
	- open the bin folder inside it and copy all of those dll files to the bin folder of your NVIDIA GPU Computing Toolkit installation.
	- The location will likely be `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
	- There's also a possibility you'll see an error later for a missing cublas64_##.dll file - if this happens, open this bin folder again and look for a dll file named similarly like `cublas64_12.dll` (assuming you got an error for missing `cublas64_11.dll`) You can create a copy of the file that does exist and just rename it to the missing one if this happens.
- it's also possible, depending on your motherboard and graphics card that you might need to change some settings in the BIOS to allow your OS to access the GPU. it's been a few months since I did that so unfortuantely I don't remember what I did, but you can definitely google it if you come across any errors related to your CUDA installation.
- With any luck, this should be all you need to get CUDA working on your PC.

## Preparing your Dataset

### Create or acquire a folder full of wav files

This is going to be different depending on what voice you're training, so it's hard to go into every possible option, but in the end you'll need a folder full of 2-10 second long wav files. If you're recording your own voice, make sure you use a good microphone, try to avoid any background noise or music, and don't leave a lot of empty space before or after the speech.

I would probably aim for at least an hour worth of audio files (possibly more). I know that seems like a lot, but you're teaching an AI model how to talk from scratch. it doesn't really know how to speak.

Your files should all have a common naming convention like follows:
```
- voice_0001.wav
- voice_0002.wav
- voice_0003.wav
```
### Convert your wav files to correct format

Assuming you have ffmpeg installed, you can accomplish that as follows:
- Open the folder where your wav files are located.
- Add a new folder named "converted" in that folder.
- Click in the address bar at the top of your wavs folder, and type `cmd` then press enter to open a command prompt at that location.
- now enter the below command in the command prompt to mass convert all of your files to 48kHz and mono:
```
for %f in (*.wav) do ffmpeg -i "%f" -ac 1 -ar 24000 -sample_fmt s16 "converted\%~nf.wav"
```
- You should now have 24kHz copies of all your wav files in the converted folder.

### Transcribe your audio

For each wav file you plan on using to train your model, you'll need the full text transcript of that audio. If you're training on your own voice, it might be easier to find a list of sentences to read, then match those up to the recordings later. You could also manually transcribe the files, or use something like OpenAI's [Whisper](https://openai.com/index/whisper/) or opensource [WhisperX](https://github.com/m-bain/whisperX) to convert your wav files to text. I started with sentences, so I didn't use any automated transcription.

### Build a metadata.csv file

While training isn't specifically going to use this file, we're going to use it to generate the required Training Data.

If you want you can do this manually with something like Notepad++ or write a quick script in whatever programming language you're comfortable with, but your file should be in the following format:

`{filename}|{Full Text Transcription of file}|0`

Here's an example:
```
voice_0001.wav|Yeah. I know nothing about AI. Wait, I can pay you if you want.|0
voice_0002.wav|I'm learning things while I'm here. Oh, I heard her.|0
voice_0003.wav|Do you want me to pay you? I'm not doing this for free.|0
etc...
```

### Setup your python environment

- in the folder where you cloned StyleTTS2, click on the address bar and type `cmd` then press enter to open a command prompt window at this location.
- Enter the following command to create a [Python Virtual Environment](https://www.geeksforgeeks.org/python/create-virtual-environment-using-venv-python/):
```
python -m venv venv
```
- This will take a little while to run and will create a folder called venv with stuff for your Virtual Environment.
- Next to start the virtual environment, enter:
```
venv\Scripts\activate
```
- you'll see `(venv)` show up to the left of your prompt, which says the Virtual Environment is started.
	- any time you need to reopen a command prompt to run python commands, you need to start the Virtual Environment again.
- Next, you'll need to install the python dependencies for this project. So, type the following command:
```
pip install -r requirements.txt
```
- This will install all of the packages listed in the requirements.txt file which will save you a lot of headache later.
- That will also install pytorch, which is the library that allows python to communicate with your GPU, but the default version is only installed to work with your CPU.
- so, next run the following couple of commands to uninstall pytorch and reinstall it with the correct CUDA compatibility:
```
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# important note: I'm currently using torch version 2.7.1 and there's a possibilility
# future versions could break this functionality, so you can install specific versions like this if needed:
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```
- There's a possibility you'll see more errors for missing python packages when you try to run training. it's usually a safe bet you can just run `pip install packagename` to fix the missing dependency.
	- I think `pandas` was one of them, so you might need to run `pip install pandas` (as an example)

### Convert metadata.csv to Training Data

- copy the `build_dataset.py` script to the base directory of your local StyleTTS2 folder
- save your `metadata.csv` file wherever you want. Mine's in the 'StyleTTS2/Data' folder
- change the 'Path/to/metadata.csv' at the top of the build_dataset file to point to the location of your file
	- keep in mind, Python uses forward slashes to navigate directories.
- At this point, if you don't already have a command prompt open from your StyleTTS2 folder, open one and start the Virtual Environment
- now type the following command to build your dataset:
```
python build_dataset.py
```
That script will randomize and convert your metadata.csv file into two files: `train_list.txt` and `val_list.txt`. You can open them to verify they're in the following format:
```
voice_0094.wav|wˌeəz ðə tˈɒɡəl wˌɒt tˈɒɡəl ɑː juː lˈʊkɪŋ fɔː|0
voice_0052.wav|nˈəʊ nˈəʊ ʃˈeə mˌiː ðə sˈiːkɹɪts kˈʌm ˈɒn|0
```
90% of the files go to train_list, and the other 10% go to val_list - also, it changes all of your texts to phonemes, which is highly recommended for training.
- Note: you Can train on plain english text, but your TTS model will likely be limited to pronunciation for words spoken in your dataset, which kind of defeats the purpose.

You'll also want to make sure your 24000 kHz wav files are copied to a folder called `wavs` inside your `StyleTTS2\Data` folder

### Prep your Config

inside the `StyleTTS2\Configs` folder, there's a config.yml file. You should open this with something like Notepad++ or if you're using an IDE, just open it in that.
Understanding/Modifying this config file is going to be important for running training.
Here are some important config settings to know:
- **log_dir**: The folder where checkpoints and logs will be saved during training.
- **first_stage_path**: This is the name of the final checkpoint file generated by train_first.py - train_second.py will load this checkpoint to continue training.
- **save_freq**: This is how many epochs run between saving checkpoint files. I think it might be possible to continue training from an existing checkpoint if training gets interrupted or fails randomly, but I haven't worked that out yet. I'll update here if I do figure it out.
- **device**: Leave this set as "cuda". Or, if you have multiple GPUs, you could configure a specific one like "cuda:0"
- **batch_size**: This controls how many sample files from your dataset training attempts to process at a time. Higher number means training might run faster, but it uses more GPU memory, so you can crash training or make it run crazy slow if you're using close to 100% of your GPU.
	- Note: for the first round of training I had to use 2 as the batch size with about 1040 wav files and `max_len: 400`
- **max_len**: (I believe) this controls the max audio length wav file training will attempt to process. You can lower this to save on GPU memory if needed, but if you go too low, you might end up with an untrained model. (200 is the lowest I've used successfully)
- **data_params**: The txt files specified here should exist and already be populated with your training data from running build_dataset.py
	- **root_path**: Make sure the corresponding wav files from your dataset are in the path specified here
	- Note: for `OOD_texts.txt` This wasn't generated from running build_dataset.py, but the one from the StyleTTS2 github repository should have already been cloned down and should work without any changes. You can create your own if you want, but it will require converting random sentences to phonemes similar to the other files. Additional Info: The wav file paths listed in `OOD_texts.txt` are ignored if they're present. so the format can either be `somefile.wav|phonemized sentence(s)|0` Or, `phonemized sentence(s)|0` - either should work.
- **preprocess_params.sr**: Leave the sr value at 24000 to match your sample wav files.
- **loss_params**:
	- **TMA_epoch**: I don't know exactly what this adds to training, but after this epoch, your first stage training will start using more memory. So, be aware, even if it seems like the first 50 epochs are running fine, it might still run out of memory around epoch 50 - in which case you might need to adjust other settings to lower memory usage.
	- **diff_epoch**: similar to TMA_epoch but for round 2 training
	- **joint_epoch**: similar to the other two settings for round 2 training
	- I believe I read somewhere if you set diff_epoch to higher than joint_epoch, then it won't run diffusion training?
	- I'm really not an expert on how the training works, just kind of know what the settings do, so I would recommend not changing any these loss params from their default values unless you've read into them and you know what changing them will accomplish.
- **slmadv_params**: settings for SLM adversarial training
	- **batch_percentage**: defaults to 0.5 percent of your file batches to use for SLM adversarial training. You can lower this to probably 0.2 if you need to lower overall memory usage.
- For everything else, I would recommend leaving the default values unless you know what changing it will do. changing some values will break training, and changing some values will prevent inference from working later without additional steps, so proceed with caution.

### Running Training

At this point, as long as everything up until this point has gone well, you should be good to start training. If you run into any errors unrelated to CUDA, this is when you can try just running `pip install packagename` to fix it. If there are any CUDA errors, you might need to google how to fix it if I didn't cover it above.

Also, since newer versions of pytorch use a "weights_only_unpickler" by default, when you try to run training, you might run into an error about weights_only being set to True/False. Your first instinct might be to try changing versions until you get something that works, but that's going to lead you into dependency hell trying to get the magic mix of versions that makes everything happy. Fortunately, there's an easy yet hacky solution to this problem.

if this error pops up, open the folder at `StyleTTS2\Lib\site-packages\torch\` and find the file `serialization.py`. Open this file and find everywhere that sets the value of `weights_only` then manually set it to False. Save the file and as long as you did it correctly, that error should be fixed. (I believe you can pass weights_only=False into any calls to that class, but it's easier in my opinion just to force it to False.)

To start the first round of training, run the following command from a command prompt in the StyleTTS2 folder (make sure your venv is running):
```
python train_first.py
```

Now, if your training config is the same as mine, it's probably going to take a while to run the first round of training. Maybe about 24 hours or more, but in the end you should end up with a folder full of checkpoint files like `epoch_1st_00195.pth` and `first_stage.pth` as well as `config.yml` which is just a copy of the config.yml file you modified in the Configs folder.

At this point, you "should" be able to run the second round of training, but there are some problems that will pop up.
- For starters, you need to set the `batch size` in the `config.yml` file to a minimum of 8. Anything less than that, and it's going to throw errors for loss values being NaN.
- Also, at the bottom of the `StyleTTS2/models.py` file, there's a `load_checkpoint` function that you might need to modify for round 2 training. I've added a `load_checkpoint.py` file with the modified implementation. You can just comment out the original one and copy/paste from that file. You'll probably need to switch back if you run round 1 training again.
- Most importantly, since you're updating batch size to 8, you're now going to be using a lot more GPU memory on top of the fact that round 2 training adds more training methods that already use more memory than round 1. As a result, I ended up renting a pod in the cloud with an 80GB GPU and ran training there.
- it's possible you could lower the `max_len` in config.yml to maybe 200 and lower the `batch_percentage` to 0.2 to force round 2 training to work on your local PC, but if you don't run out of memory, it's going to take a very long time to complete, and it might still fail in later epochs when other types of training start.
- anyway, After you've run the 2nd round of training and you have an `epoch_2nd_000##.pth` file you'll be able to run inference on your trained model and finally hear what it sounds like.

## Inference

This was probably the most difficult part for me. There is an ipynb in the StyleTTS2 demo folder that shows how to run inference on LJSpeech, but it heavily depends on the steps you took above during training. If you change the wrong thing, good luck ever using your checkpoint files without learning how everything works. :/ Also, if you don't phonemize the texts in your dataset before training, the sample ipynb files either won't work, or will result in garbled nonsense.

Important point: It's good to understand the difference between the LibriTTS and LJSpeech inference examples. LibriTTS is the base model that LJSpeech was finetuned from, so inference works differently between the two methods.

I'm going to include a basic `inference.py` script with `tts.py` you can use to test your models as long as you followed the instructions above. Important notes:
- You should copy both of those files into your base StyleTTS2 folder because they rely on other files in the StyleTTS2 project.
- you'll need to modify the constants at the top of `inference.py` to match your file structure.
- Make sure you use the same config.yml file that you used during training.
- DEFAULT_TARGET_VOICE_PATH: this is the sample file you want StyleTTS2 to base your generated voice's style on.
	- Important note about StyleTTS2: it requires you to provide a sample wav file during inference to use as a style reference. This can affect the speed, tone and other characteristics about the resulting voice. You can switch between files to change the Style and speed of your TTS voice.
- You may run into a few dependencies during inference that aren't already installed in your project because we didn't use them during training. The imports at the top of tts.py are going to be the most likely things missing, so just `pip install packagename` if any errors pop up.

once you have everything copied to your StyleTTS2 root directory and you've updated the inference.py script, just run the following command:
```
python inference.py
```

If everything went as expected, you should have now run inference and generated a wav file. Congratulations! It's really not easy getting to this point, and there's so much conflicting information spread out across different places on the internet, even popular AI models struggle to give correct answers on the subject.

If there's any interest, I could possibly document the process of running `train_second.py` on RunPod, but it can be stressful and costs money while the pod is running, so I don't know if I want that on my conscience where people are depending on timely answers to their problems.

Anyway, best of luck, and kudos if you managed to get this far. I went through a hell of a lot of trouble to get all of that figured out. If you feel like anything I said was wrong or unclear, please feel free to point it out. I'm not a python developer, I'm not a Linux user, and I know very little about the science behind training AI models.
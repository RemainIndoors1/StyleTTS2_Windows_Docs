
import os
import random
import subprocess
import unicodedata

# *** Change this to your metadata.csv file path *** 
input_file = 'Path/to/metadata.csv'

def phonemize_with_espeak(text):
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

def phonemize_line(line):
    parts = line.strip().split('|')
    if len(parts) != 3:
        return None
    wav, text, speaker_id = parts
    phonemes = phonemize_with_espeak(text)
    return f"{wav}|{phonemes}|{speaker_id}"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if "|" in line]
    split_idx = int(len(lines) * 0.9)
    random.shuffle(lines)

with open("Data/train_list.txt", "w", encoding="utf-8") as f:
    for line in lines[:split_idx]:
        phonemized = phonemize_line(line)
        f.write(f"{phonemized}\n")

with open("Data/val_list.txt", "w", encoding="utf-8") as f:
    for line in lines[split_idx:]:
        phonemized = phonemize_line(line)
        f.write(f"{phonemized}\n")
    
"""
Generates mel spectrogram PNGs for every .wav file in Seperation/.
Reference for each (experiment group, instrument) is set from the original stem,
so generated stems are always compared on the same scale as the ground truth.
Output: same folder, same filename but .png
Run: conda run -n ctm_gen python gen_spectrograms.py
"""

import os, glob, re
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEPERATION_ROOT = os.path.join(os.path.dirname(__file__), 'Seperation')

def get_group(wav_path):
    folder = os.path.basename(os.path.dirname(wav_path))
    return 'Slakh' if folder.startswith('Slakh') else 'MUSDB'

def get_instrument(wav_path):
    name = os.path.splitext(os.path.basename(wav_path))[0]
    if 'Mixture_audio' in name:
        return 'mix'
    m = re.search(r'(?:original|generated)_(\w+)_\d+', name)
    return m.group(1) if m else 'unknown'

wav_files = glob.glob(os.path.join(SEPERATION_ROOT, '**', '*.wav'), recursive=True)
wav_files.sort()
print(f"Found {len(wav_files)} wav files")

# Pass 1: compute all spectrograms; record max from original folders only
print("Pass 1: computing spectrograms...")
cache = {}
orig_ref = {}  # (group, instrument) -> max power from original stem

for wav_path in wav_files:
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=256,
        n_mels=128, fmax=sr // 2, power=2.0,
    )
    cache[wav_path] = (S, sr)
    folder = os.path.basename(os.path.dirname(wav_path))
    if 'original' in folder.lower():
        key = (get_group(wav_path), get_instrument(wav_path))
        orig_ref[key] = max(orig_ref.get(key, 0.0), float(S.max()))

print("References (from originals):")
for k, v in sorted(orig_ref.items()):
    print(f"  {k[0]:6s}  {k[1]:8s}  {v:.2f}")

# Pass 2: render using original-derived reference per (group, instrument)
print("Pass 2: saving PNGs...")
for wav_path in wav_files:
    S, sr = cache[wav_path]
    key = (get_group(wav_path), get_instrument(wav_path))
    ref = orig_ref.get(key, float(S.max()))

    S_db = librosa.power_to_db(S, ref=ref)
    fig, ax = plt.subplots(figsize=(5, 1.4))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=256,
        x_axis=None, y_axis=None,
        cmap='magma', ax=ax,
        vmin=-80, vmax=0,
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path = os.path.splitext(wav_path)[0] + '.png'
    fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"  {os.path.relpath(wav_path, SEPERATION_ROOT)}")

print("Done.")

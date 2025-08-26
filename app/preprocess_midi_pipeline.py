'''
# MIDI → GAN Preprocessing Pipeline

This Python File converts raw MIDI files (metal/classical) into **GAN-ready piano-roll tensors** with a consistent beat grid, plus a CSV of metadata for each training window.

## What it does (high level)

1. **Validate**: Skips corrupt/too-short MIDI files.
2. **Extract timing**: Reads tempo map & time signatures.
3. **Normalize to beats**: Converts note times into **beat-relative grid** (tempo-invariant).
4. **Quantize**: Snaps to a fixed subdivision (steps per beat).
5. **Instrument mapping**: Groups tracks into fixed channels (e.g., guitar/bass/drums for metal).
6. **Transpose (augmentation)**: Shifts pitches by semitones (drums untouched).
7. **Slice windows**: Cuts into fixed-length bar windows with overlap.
8. **Save**: Writes each window as a `.npy` tensor and logs metadata.

## Output format

Each saved numpy array has shape:

```
[C, T, P]
C = instrument channels (metal: 4, classical: 5)
T = time steps (bars * 4 * steps_per_beat)
P = pitches (pitch_high - pitch_low + 1)  # default 21..108
```

Binary values (0/1) indicate note presence at a time-step and pitch.


## Usage

**Metal** (4 bars, 16th-note grid, ±5 semitone transpose):
```bash
python preprocess_midi_pipeline.py \
  --genre metal \
  --raw_dir data/midi/metal/raw\
  --out_dir data/midi/metal/processed \
  --transpose -2 0 2 \
  --bars 4 \
  --hop_bars 2 \
  --steps_per_beat 4 \
  --pitch_low 36 \
  --pitch_high 84 \
  --min_duration_sec 10.0
```

**Classical** (8 bars, 32nd-note grid, full octave transpose):
```bash
python preprocess_midi_pipeline.py \
  --genre classical \
  --raw_dir data/midi/classical/raw \
  --out_dir data/midi/classical/processed \
  --transpose -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 \
  --bars 8 --hop_bars 4 \
  --steps_per_beat 8 \
  --min_duration_sec 20
```

## Theory (why each step matters)

- **Beat-relative grid**: Rhythms become tempo-invariant (same pattern at 90 or 180 BPM looks identical on the grid).
- **Quantization**: Reduces micro-timing noise; GANs learn cleaner rhythmic patterns.
- **Instrument channels**: Lets the model learn cross-instrument interaction (e.g., guitar vs. drums).
- **Transposition**: Teaches interval patterns and multiplies dataset without overfitting to a key.
- **Fixed windows**: GANs need fixed-size tensors; overlapping windows increase data and keep continuity.

## Notes & limits

- Tempo changes are captured as metadata; slicing assumes 4/4 for window size. For mixed meters, adapt `steps_per_bar` using time signature.
- Drum handling is "single-channel" by default; you can extend to multi-kit channels (kick/snare/hihat).
- If you need velocity (continuous) instead of binary rolls, extend `build_pianoroll` to store normalized velocities.

## Next steps

- Build a PyTorch `Dataset` that loads these `.npy` tensors and yields batches.
- Train **separate GANs** for metal and classical.
- Optionally condition the GAN on metadata (tempo, transpose, instrument presence).

'''
#argparse library in Python is used for parsing command-line argument
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm



#Python package for working with MIDI files (Musical Instrument Digital Interface
import pretty_midi


# -----------------------------
# Configs
# -----------------------------

GENRE_CONFIGS = {
    "metal": {
        "pitch_low": 21,    # A0
        "pitch_high": 108,  # C8
        "drum_mode": "single",  # "single" or "kit3" (kick/snare/hihat)
        "steps_per_beat": 4,    # 16th-note grid
        "bars": 4,
        "hop_bars": 2,
        "transpose": [-5, -3, 0, 3, 5],
        "min_duration_sec": 8.0,
    },
    "classical": {
        "pitch_low": 21,
        "pitch_high": 108,
        "drum_mode": "single",
        "steps_per_beat": 8,    # 32nd-note grid
        "bars": 8,
        "hop_bars": 4,
        "transpose": list(range(-12, 13)),  # full octave
        "min_duration_sec": 20.0,
    },
}


# -----------------------------
# Utilities
# -----------------------------

def is_valid_midi(path: Path, min_duration_sec: float) -> bool:
    '''
    Validate raw MIDI :
    What: Skip corrupt/too-short files.
    Why (theory): Bad inputs increase noise and hurt GAN convergence 
    (discriminator gets easy negatives; generator overfits artifacts)
    '''
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
        return pm.get_end_time() >= min_duration_sec
    except Exception:
        return False


def get_tempo_map(pm: "pretty_midi.PrettyMIDI") -> Tuple[np.ndarray, np.ndarray]:
    '''
    Extract tempo & meter :
    What: Read tempo changes and time signatures.
    Why: We will work on a beat grid; knowing tempo lets us map seconds → beats consistently. 
    Meter (e.g., 4/4) guides window sizes.
    '''
    times, tempos = pm.get_tempo_changes()
    if len(tempos) == 0:
        # pretty_midi returns empty only in rare malformed files; default 120 BPM
        tempos = np.array([120.0], dtype=float)
        times = np.array([0.0], dtype=float)
    return times, tempos 

    #times → times (in seconds) when tempo changes occur.
    #tempos → tempo (BPM) at those times.
    #Handles rare case where MIDI has no tempo info by defaulting to 120 BPM.


'''------------------------------------------------------------------------------------------------------------------------------------'''
'''
Normalize time to a beat grid:

What: Convert every note’s start/end time in seconds to beats, then to discrete grid steps.
Why: Beat-relative grids make rhythms tempo-invariant; the same riff at 90 vs 180 BPM has the same pattern.

'''

def time_to_beat_index(pm: "pretty_midi.PrettyMIDI", t: float) -> float:
    """Map a time in seconds to a continuous beat index using the beat times array."""
    beat_times = pm.get_beats()
    if len(beat_times) == 0:
        # Fallback: approximate using first tempo
        times, tempos  = get_tempo_map(pm)
        tempo = tempos[0]
        sec_per_beat = 60.0 / tempo
        return t / sec_per_beat
    beat_idx = np.interp(t, beat_times, np.arange(len(beat_times), dtype=float))
    return float(beat_idx)
    #times: np.ndarray of times (seconds) of tempo changes
    #tempos: np.ndarray of tempo values (BPM) at those times


def quantize_note_to_grid(pm: "pretty_midi.PrettyMIDI", start_s: float, end_s: float, steps_per_beat: int) -> Tuple[int, int]:
    """Quantize a note's start/end times (seconds) into integer grid steps in beat space."""
    s_b = time_to_beat_index(pm, start_s)
    e_b = time_to_beat_index(pm, end_s)
    s_step = int(round(s_b * steps_per_beat))
    e_step = int(round(e_b * steps_per_beat))
    if e_step <= s_step:
        e_step = s_step + 1  # ensure at least 1 step
    return s_step, e_step

'''------------------------------------------------------------------------------------------------------------------------------------'''

def map_instrument_channel(inst: "pretty_midi.Instrument", genre: str, drum_mode: str) -> Optional[int]:
    """
    Map pretty_midi Instrument to a fixed channel index.
    Returns:
        channel index int or None to ignore.
    Channel mapping proposal:
        metal: 0=guitar, 1=bass, 2=drums, 3=other
        classical: 0=piano/keyboard, 1=strings, 2=winds, 3=brass, 4=percussion
    """
    if genre == "metal":
        if inst.is_drum:
            return 2
        prog = inst.program  # 0-127
        # rough mapping by GM program families
        if 24 <= prog <= 31:  # Guitar family
            return 0
        if 32 <= prog <= 39:  # Bass family
            return 1
        # Keys/Leads/Strings as other accompaniment
        return 3
    else:  # classical
        if inst.is_drum:
            return 4  # percussion
        prog = inst.program
        if 0 <= prog <= 7:   # Piano
            return 0
        if 40 <= prog <= 47: # Strings (Violin…)
            return 1
        if 56 <= prog <= 63: # Brass
            return 3
        if 64 <= prog <= 71: # Ensemble/Choir
            return 1
        if 72 <= prog <= 79: # Woodwinds
            return 2
        # Harp/Celesta etc. -> map to piano/strings group
        if 8 <= prog <= 15:
            return 0
        if 46 <= prog <= 54:
            return 1
        # default other -> strings group to avoid losing too much
        return 1


def build_pianoroll(pm: "pretty_midi.PrettyMIDI",
                    genre: str,
                    steps_per_beat: int,
                    pitch_low: int,
                    pitch_high: int,
                    drum_mode: str) -> Tuple[np.ndarray, Dict]:
    """
    Build a multi-channel piano-roll tensor from a MIDI, aligned to a beat grid.
    Returns:
        roll: np.ndarray [channels, T, P] binary
        info: dict with time_signatures, tempos, length, etc.
    """
    # compute total length in steps using last note end
    end_s = pm.get_end_time()
    # Approximate beat length to size the grid
    last_step = int(round(time_to_beat_index(pm, end_s) * steps_per_beat)) + 1
    P = (pitch_high - pitch_low + 1)

    # Number of channels by genre
    num_channels = 4 if genre == "metal" else 5
    roll = np.zeros((num_channels, last_step, P), dtype=np.uint8)

    for inst in pm.instruments:
        ch = map_instrument_channel(inst, genre, drum_mode)
        if ch is None:
            continue
        if inst.is_drum:
            # simple: mark any drum hit at the time step (no pitch dimension) -> use middle pitch bin as marker
            for n in inst.notes:
                s, e = quantize_note_to_grid(pm, n.start, n.end, steps_per_beat)
                p_idx = min(max(n.pitch - pitch_low, 0), P - 1)
                roll[ch, s:e, p_idx] = 1
        else:
            for n in inst.notes:
                p = n.pitch
                if p < pitch_low or p > pitch_high:
                    continue
                s, e = quantize_note_to_grid(pm, n.start, n.end, steps_per_beat)
                roll[ch, s:e, p - pitch_low] = 1

    tempo_times, tempos  = get_tempo_map(pm)
    tsigs = [(ts.numerator, ts.denominator, ts.time) for ts in pm.time_signature_changes]
    info = {
        "tempos": tempos.tolist(),
        "tempo_times": tempo_times.tolist(),
        "time_signatures": tsigs,
        "last_step": int(last_step),
        "steps_per_beat": int(steps_per_beat),
        "pitch_low": int(pitch_low),
        "pitch_high": int(pitch_high),
        "num_channels": int(num_channels),
    }
    return roll, info

'''------------------------------------------------------------------------------------------------------------------------------------'''

def apply_transpose_roll(roll: np.ndarray, shift: int, pitch_low: int, pitch_high: int) -> np.ndarray:
    """Transpose a [C, T, P] piano-roll by shifting the pitch axis. Drums/percussion channels should not be shifted."""
    C, T, P = roll.shape
    out = np.zeros_like(roll)
    for c in range(C):
        # Heuristic: treat highest channel index as percussion for classical; for metal, channel 2 is drums
        is_drum_ch = (c == 2)  # metal default
        # Adjust for classical 5 channels (percussion index 4)
        if C == 5 and c == 4:
            is_drum_ch = True
        if is_drum_ch:
            out[c] = roll[c]  # no transpose
            continue
        if shift == 0:
            out[c] = roll[c]
            continue
        if shift > 0:
            out[c, :, shift:] = roll[c, :, :P-shift]
        else:
            out[c, :, :P+shift] = roll[c, :, -shift:]
    return out


def slice_windows(roll: np.ndarray, steps_per_bar: int, bars: int, hop_bars: int) -> List[np.ndarray]:
    """
    Slice a [C, T, P] roll into fixed-length windows of length bars*steps_per_bar.
    """
    C, T, P = roll.shape
    win = bars * steps_per_bar
    hop = hop_bars * steps_per_bar
    if win <= 0 or hop <= 0:
        return []
    windows = []
    for s in range(0, max(T - win + 1, 0), hop):
        windows.append(roll[:, s:s+win, :])
    return windows


def velocity_jitter(x: np.ndarray, prob: float = 0.0) -> np.ndarray:
    """
    Placeholder for velocity/intensity augmentation. With binary rolls this is a no-op.
    """
    return x


def save_samples_and_metadata(samples: List[np.ndarray], meta_rows: List[Dict], out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    arr_dir = out_dir / "arrays"
    arr_dir.mkdir(exist_ok=True)
    meta = pd.DataFrame(meta_rows)
    for i, arr in enumerate(samples):
        np.save(arr_dir / f"{prefix}_{i:06d}.npy", arr)
    meta_path = out_dir / f"{prefix}_metadata.csv"
    meta.to_csv(meta_path, index=False)
    return meta_path


def process_file(path: Path,
                 genre: str,
                 cfg: Dict,
                 transpose_list: List[int],
                 bars: int,
                 hop_bars: int,
                 steps_per_beat: int,
                 pitch_low: int,
                 pitch_high: int) -> Tuple[List[np.ndarray], List[Dict]]:

    pm = pretty_midi.PrettyMIDI(str(path))
    roll, info = build_pianoroll(pm, genre, steps_per_beat, pitch_low, pitch_high, cfg.get("drum_mode", "single"))
    steps_per_bar = steps_per_beat * 4  # assume 4/4 for slicing; advanced: compute from time signature

    samples: List[np.ndarray] = []
    meta_rows: List[Dict] = []

    for shift in transpose_list:
        tr = apply_transpose_roll(roll, shift, pitch_low, pitch_high)
        wins = slice_windows(tr, steps_per_bar, bars, hop_bars)
        for widx, w in enumerate(wins):
            w = velocity_jitter(w, prob=0.0)
            if w.sum() == 0:
                continue  # skip empty
            samples.append(w.astype(np.uint8))
            meta_rows.append({
                "src_file": str(path),
                "genre": genre,
                "transpose": shift,
                "window_index": int(widx),
                "bars": int(bars),
                "steps_per_beat": int(steps_per_beat),
                "pitch_low": int(pitch_low),
                "pitch_high": int(pitch_high),
                "channels": int(w.shape[0]),
                "time_steps": int(w.shape[1]),
                "pitches": int(w.shape[2]),
                "notes_count": int(w.sum()),
                "tempos_json": json.dumps(info["tempos"]),
                "tempo_times_json": json.dumps(info["tempo_times"]),
                "time_signatures_json": json.dumps(info["time_signatures"]),
            })
    return samples, meta_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genre", choices=list(GENRE_CONFIGS.keys()), required=True)
    ap.add_argument("--raw_dir", type=str, required=True, help="Folder with raw .mid/.midi files")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for processed arrays + metadata")
    ap.add_argument("--transpose", type=int, nargs="+", default=None, help="List of semitone shifts to apply")
    ap.add_argument("--bars", type=int, default=None)
    ap.add_argument("--hop_bars", type=int, default=None)
    ap.add_argument("--steps_per_beat", type=int, default=None)
    ap.add_argument("--pitch_low", type=int, default=None)
    ap.add_argument("--pitch_high", type=int, default=None)
    ap.add_argument("--min_duration_sec", type=float, default=None)
    args = ap.parse_args()

    cfg = GENRE_CONFIGS[args.genre].copy()
    if args.transpose is not None: cfg["transpose"] = args.transpose
    if args.bars is not None: cfg["bars"] = args.bars
    if args.hop_bars is not None: cfg["hop_bars"] = args.hop_bars
    if args.steps_per_beat is not None: cfg["steps_per_beat"] = args.steps_per_beat
    if args.pitch_low is not None: cfg["pitch_low"] = args.pitch_low
    if args.pitch_high is not None: cfg["pitch_high"] = args.pitch_high
    if args.min_duration_sec is not None: cfg["min_duration_sec"] = args.min_duration_sec

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_files = [p for p in raw_dir.rglob("*") if p.suffix.lower() in (".mid", ".midi")]
    if not midi_files:
        print(f"No MIDI files found in {raw_dir}")
        return

    samples_all: List[np.ndarray] = []
    meta_all: List[Dict] = []

    for p in tqdm(midi_files, desc=f"Processing {args.genre} MIDIs"):
        if not is_valid_midi(p, cfg["min_duration_sec"]):
            continue
        try:
            samples, meta = process_file(
                path=p,
                genre=args.genre,
                cfg=cfg,
                transpose_list=cfg["transpose"],
                bars=cfg["bars"],
                hop_bars=cfg["hop_bars"],
                steps_per_beat=cfg["steps_per_beat"],
                pitch_low=cfg["pitch_low"],
                pitch_high=cfg["pitch_high"],
            )
            samples_all.extend(samples)
            meta_all.extend(meta)
        except Exception as e:
            # skip problematic file
            print(f"[WARN] Skipping {p.name}: {e}")

    # Save arrays individually + single metadata CSV
    arr_dir = out_dir / "arrays"
    arr_dir.mkdir(exist_ok=True)
    meta_df = pd.DataFrame(meta_all)
    for i, arr in enumerate(samples_all):
        np.save(arr_dir / f"{args.genre}_{i:07d}.npy", arr)
    meta_df.to_csv(out_dir / f"{args.genre}_metadata.csv", index=False)

    print(f"Saved {len(samples_all)} windows to {arr_dir}")
    print(f"Metadata: {out_dir / (args.genre + '_metadata.csv')}")


if __name__ == "__main__":
    main()

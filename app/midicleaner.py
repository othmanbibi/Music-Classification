import pretty_midi
import os


def is_valid_midi(path, min_duration=8.0):
    try:
        pm = pretty_midi.PrettyMIDI(path)
        return pm.get_end_time() >= min_duration
    except Exception:
        return False


raw_dir_metal = "data/midi/metal/raw"
clean_dir_metal = "data/midi/metal/cleaned"

os.makedirs(clean_dir_metal, exist_ok=True)



#Metal min duration: 8–15s (to capture riffs)
for file in os.listdir(raw_dir_metal):
    if file.endswith(".mid") and is_valid_midi(os.path.join(raw_dir_metal, file), min_duration=15.0):
        os.rename(os.path.join(raw_dir_metal, file), os.path.join(clean_dir_metal, file))





raw_dir_classical = "data/midi/classical/raw"
clean_dir_classical = "data/midi/classical/cleaned"

os.makedirs(clean_dir_classical, exist_ok=True)

#Classical min duration: 20–30s (for phrases)
for file in os.listdir(raw_dir_classical):
    if file.endswith(".mid") and is_valid_midi(os.path.join(raw_dir_classical, file), min_duration=30.0):
        os.rename(os.path.join(raw_dir_classical, file), os.path.join(clean_dir_classical, file))
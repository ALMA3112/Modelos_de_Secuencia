"""Conversion de matrices (T,4) a MIDI con pretty_midi."""
import numpy as np
import pretty_midi


def step_seconds(tempo):
    return 60.0 / (4 * tempo)


def voice_runs(col):
    """[(inicio, fin, nota)] fusionando pasos consecutivos iguales. 0 = silencio (se omite)."""
    col = [int(c) for c in col]
    runs, i, n = [], 0, len(col)
    while i < n:
        j = i
        while j + 1 < n and col[j + 1] == col[i]:
            j += 1
        if col[i] != 0:
            runs.append((i, j + 1, col[i]))
        i = j + 1
    return runs


def to_midi(arr, tempo=120, path=None, context_steps=32):
    arr = np.asarray(arr)
    dt = step_seconds(tempo)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for v in range(arr.shape[1]):
        inst = pretty_midi.Instrument(program=0, name=f"voz{v}")
        for s, e, p in voice_runs(arr[:, v]):
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=p,
                                               start=s * dt, end=e * dt))
        pm.instruments.append(inst)
    pm.lyrics.append(pretty_midi.Lyric("inicio de la continuacion", context_steps * dt))
    if path is not None:
        pm.write(str(path))
    return pm

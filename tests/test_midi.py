import numpy as np
import pretty_midi

from bach.midi import step_seconds, to_midi, voice_runs


def test_paso_dura_0125_a_120_bpm():
    assert step_seconds(120) == 0.125


def test_fusion_y_silencio():
    assert voice_runs([60, 60, 0, 0, 62, 60, 60]) == [(0, 2, 60), (4, 5, 62), (5, 7, 60)]


def test_midi_cuatro_voces(tmp_path):
    arr = np.array([[60, 64, 67, 72]] * 4 + [[0, 0, 0, 0]] * 2)
    p = tmp_path / "a.mid"
    pm = to_midi(arr, 120, p, context_steps=2)
    assert len(pm.instruments) == 4
    notas = [n for i in pm.instruments for n in i.notes]
    assert len(notas) == 4
    assert all(abs(n.end - n.start - 0.5) < 1e-6 for n in notas)
    back = pretty_midi.PrettyMIDI(str(p))
    assert len(back.instruments) == 4 and abs(back.get_end_time() - 0.5) < 0.01

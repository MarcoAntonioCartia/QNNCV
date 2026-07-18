"""
Weights persistence for CVQGANGenerator (best_weights.npy).

Two on-disk formats, same filename:

v1 (legacy, input_displacement): a raw flat float array — exactly
    np.save(path, weights). Byte-compatible with every pre-existing
    best_weights.npy; loads WITHOUT allow_pickle.

v2 (interleaved_squeeze): a pickled dict (0-d object array) carrying the
    flat weights, the (n_layers, n_modes) w_s modulation amplitudes, and
    enough config metadata to sanity-check a reload:
        {'format': 'qnncv-weights-v2', 'encoding': 'interleaved_squeeze',
         'weights', 'w_s', 'n_modes', 'n_layers', 'cutoff_dim', 'use_kerr',
         'inj_layers', 'r_max'}

Compatibility contract:
- v1 file -> interleaved generator: weights assigned, w_s left at zero
  (warm start from the frozen baseline); a notice is printed.
- v2 file -> input_displacement generator: ValueError (the w_s amplitudes
  would be silently dropped).
"""

import numpy as np

FORMAT_V2 = 'qnncv-weights-v2'


def save_weights(path, weights, w_s=None, meta=None):
    """Save weights; v1 raw array when w_s is None, else v2 dict."""
    weights = np.asarray(weights)
    if w_s is None:
        # Legacy path: byte-identical to np.save(path, weights).
        np.save(path, weights)
        return
    payload = {
        'format': FORMAT_V2,
        'encoding': 'interleaved_squeeze',
        'weights': weights,
        'w_s': np.asarray(w_s),
    }
    if meta:
        payload.update(meta)
    np.save(path, np.array(payload, dtype=object), allow_pickle=True)


def load_weights(path):
    """Load either format -> {'weights', 'w_s' (or None), 'meta' (dict)}."""
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        payload = arr.item()
        if not (isinstance(payload, dict)
                and payload.get('format') == FORMAT_V2):
            raise ValueError(
                f"{path}: unrecognized weights payload "
                f"(expected a {FORMAT_V2} dict)")
        meta = {k: v for k, v in payload.items()
                if k not in ('weights', 'w_s')}
        return {'weights': np.asarray(payload['weights']),
                'w_s': np.asarray(payload['w_s']),
                'meta': meta}
    return {'weights': arr, 'w_s': None, 'meta': {}}


def apply_weights(generator, loaded, source='<weights file>'):
    """Assign a load_weights() result to a generator, enforcing the
    format/encoding compatibility contract."""
    weights = loaded['weights']
    w_s = loaded['w_s']

    if w_s is not None and generator.w_s is None:
        raise ValueError(
            f"{source} was saved by an interleaved_squeeze model (contains "
            f"w_s); this generator uses 'input_displacement'. Rerun with "
            f"--encoding interleaved_squeeze to use it.")

    expected = int(generator.weights.shape[0])
    if weights.shape != (expected,):
        raise ValueError(
            f"{source}: weights shape {weights.shape} does not match the "
            f"generator's ({expected},) — check n_modes/n_layers/kerr config")
    generator.weights.assign(weights)

    if generator.w_s is not None:
        if w_s is None:
            print(f"  Warm start: {source} is a legacy flat-weights file; "
                  f"w_s initialized to 0 (frozen-baseline start)")
        else:
            ws_shape = tuple(generator.w_s.shape)
            if tuple(w_s.shape) != ws_shape:
                raise ValueError(
                    f"{source}: w_s shape {tuple(w_s.shape)} does not match "
                    f"the generator's {ws_shape}")
            generator.w_s.assign(w_s)

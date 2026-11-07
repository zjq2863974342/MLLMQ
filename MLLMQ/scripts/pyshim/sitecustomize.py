"""Shim for DPG-Bench scoring on hosts without the OFA audio stack.

modelscope's `models.multi_modal.ofa_for_all_tasks` eagerly imports
`preprocessors/ofa/__init__.py`, whose first line is `from .asr import
OfaASRPreprocessor`. `asr.py` needs librosa / soundfile / fairseq -- all only
used by the OFA **ASR** (speech) preprocessor. The DPG-Bench mplug VQA path
never touches any of them.

On the original scoring host this was handled by an in-place site-packages
patch that logged:

    [multi_modal] SKIP OFA preprocessor (requires fairseq, not needed for DPG): ...

Rather than mutate site-packages, we pre-register a stand-in for the
`modelscope.preprocessors.ofa.asr` submodule (so `asr.py`'s real source, and
therefore librosa/soundfile, never runs) plus empty stand-ins for `fairseq*`.

Deliberately NOT stubbed: librosa / soundfile. `datasets` probes those with
importlib.find_spec(...) at import time, so faking them breaks datasets'
config. Leaving them truly absent is correct -- nothing on the VQA path uses
them.

`OfaASRPreprocessor` is only referenced from the ASR pipeline; the stub class
is never instantiated here. Only active when this directory is on PYTHONPATH.
"""

import importlib.machinery
import os
import sys
import types


class _Stub(object):
    """Placeholder for objects only reachable from the unsupported ASR path."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return None

    def __getattr__(self, name):
        return _Stub()

    def __iter__(self):
        return iter(())


def _stub_module(name):
    mod = types.ModuleType(name)
    fake = os.path.join("/tmp", "pyshim_" + name.replace(".", "_") + ".py")
    try:
        open(fake, "a").close()
    except OSError:
        pass
    mod.__file__ = fake
    mod.__path__ = []
    mod.__loader__ = None
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    # PEP 562: lets `from fairseq.modules import LayerNorm` and friends resolve
    # to a stub instead of raising ImportError.
    mod.__getattr__ = lambda attr: _Stub()
    sys.modules[name] = mod
    parent, _, leaf = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], leaf, mod)
    return mod


def _install():
    # 1) Short-circuit `preprocessors/ofa/asr.py` (librosa/soundfile/fairseq).
    asr = "modelscope.preprocessors.ofa.asr"
    if asr not in sys.modules:
        mod = _stub_module(asr)
        mod.OfaASRPreprocessor = _Stub

    # 2) fairseq: imported by ofa/modeling_mmspeech.py, vit.py and the
    #    generate/* modules. Also speech-only; never touched by mplug VQA.
    for name in (
            "fairseq",
            "fairseq.data",
            "fairseq.data.data_utils",
            "fairseq.data.audio",
            "fairseq.data.audio.feature_transforms",
            "fairseq.data.audio.speech_to_text_dataset",
            "fairseq.models",
            "fairseq.models.wav2vec",
            "fairseq.models.wav2vec.wav2vec2",
            "fairseq.modules",
            "fairseq.modules.transformer_sentence_encoder",
            "fairseq.utils",
    ):
        if name not in sys.modules:
            _stub_module(name)


try:
    _install()
except Exception:  # never let the shim itself break the interpreter
    pass
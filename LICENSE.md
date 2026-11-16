# License and release notes

## MIT License

Copyright (c) 2026 BASK-SR authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Scope and third-party materials

The MIT License above applies to the original BASK-SR source code and accompanying documentation released by its copyright holders. It does not replace the licenses of third-party code, libraries, datasets, or externally obtained models. Preserve applicable third-party copyright and license notices when redistributing such materials.

This project depends on PyTorch, NumPy, SciPy, SoundFile, PyYAML, and pandas. These packages and their dependencies remain subject to their respective licenses. Install them through `requirements.txt`; their inclusion as dependencies does not transfer their copyright to this project.

AEC Challenge datasets are distributed separately under their original terms. Obtain the required data and review the dataset licenses at:
https://github.com/microsoft/AEC-Challenge#dataset-licenses

The software license above does not itself grant rights to datasets or model weights. A separate licensing decision for the bundled checkpoints must be recorded before redistributing those weights as licensed open-source assets.

## Installation and inference

Install Python and a suitable PyTorch build, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python SGDRP/infer_sgdrp.py --farend farend.wav --mic microphone.wav --output enhanced.wav --device cpu
```

Inputs should be time-aligned mono WAV files sampled at 16 kHz. Use `--device cuda` for GPU inference, or add `--bypass` to obtain the BASKF output with postfilter correction bypassed. The bypass option still loads the postfilter checkpoint. Inputs of unequal length are truncated to the shorter length.

## Release scope and reproducibility

This directory provides model implementations, configurations, pretrained checkpoints, and an offline inference entry point. Complete training and paper evaluation pipelines are not included. Reproducing the paper results also requires the original data selections, evaluation scripts, checkpoint settings, delay alignment, and double-talk protection protocol. The inference command alone does not reproduce the reported tables or recovery curves.

The default checkpoints are `BASKF/weights/baskf.pt` and `SGDRP/weights/sgdrp.pt`. Alternative checkpoints may have different deployment settings. A validated dependency environment and a complete verification of third-party source provenance have not yet been supplied.

## Academic attribution

Please cite the BASK-SR paper when using this work in academic research. Add the final bibliographic entry after publication. This request does not add a condition to the MIT License.

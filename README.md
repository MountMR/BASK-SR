# Description

Lightweight acoustic echo cancellation (AEC) requires effective echo suppression, near-end speech preservation, and rapid adaptation to time-varying echo paths. This paper presents bounded adaptive subband Kalman filtering with scene-gated residual suppression (BASK-SR), a hybrid architecture combining a bounded adaptive subband Kalman filter (BASKF) with a scene-gated dual route residual postfilter (SGDRP). BASKF applies bounded neural-statistics control to analytical Kalman adaptation, while SGDRP adapts residual echo suppression to far-end single-talk (FST) and double-talk (DT) conditions. BASKF and SGDRP contain 4.989 K and 6.669 K trainable parameters, respectively. On AEC Challenge data, BASK-SR achieves 27.887 dB FST echo return loss enhancement (ERLE) and 10.206 dB DT scale-invariant signal-to-distortion ratio improvement (SI-SDRi), surpassing mainstream lightweight AEC baselines. Moreover, echo-path-change experiments demonstrate its effective recovery when the echo path changes abruptly. Overall, BASK-SR provides effective echo suppression for real-time full-duplex speech systems.

## Repository structure

```text
BASK-SR/
├── BASKF/
│   ├── baskf/                 Backbone implementation
│   ├── weights/               Backbone and initialization checkpoints
│   └── config_baskf.yaml      Backbone configuration
├── SGDRP/
│   ├── sgdrp/                 Postfilter implementation and system runtime
│   ├── weights/               Postfilter checkpoints
│   ├── config_sgdrp.yaml      Postfilter configuration
│   └── infer_sgdrp.py         Inference entry point
├── requirements.txt           Python dependencies
└── README.md
```

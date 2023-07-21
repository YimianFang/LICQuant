# LICQuant

## LSQplus_yKL

__BaseLine__: LIC scalehyperprior model 

__Quantization methods__: LSQ+, QAT

__Improvement__: minimize KLDivLoss between rounded $y_{hat}$ of the quantized and unquantized encoder

__Some Tips__:
1. Weight per-channel offsets are not recommended, which results in low computational efficiency during practical deployment.
2. Activation per-channel offsets could be implemented during integerization, but activation per-channel scales could not.

Code reference:  https://github.com/ProhetTeam/QuanTransformer
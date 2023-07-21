# LICQuant

## LSQplus_yKL
* Code reference:  https://github.com/ProhetTeam/QuanTransformer

BaseLine: LIC scalehyperprior model 

Quantization methods: LSQ+, QAT

Some Tips
1. Weight per-channel offsets are not recommended, which results in low computational efficiency during practical deployment.
2. Activation per-channel offsets could be implemented during integerization, but activation per-channel scales could not.

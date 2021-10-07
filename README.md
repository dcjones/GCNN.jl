
Implementation of a Group CNN (G-CNN) layer for Flux, which is kind of 2d
convolution operation invariant under (a discretized set of) rotations. It's
proposed in these papers.

  > Bekkers,E.J., Lafarge,M.W., Veta,M., Eppenhof,K.A.J., Pluim,J.P.W. and Duits,R. (2018) Roto-Translation Covariant Convolutional Networks for Medical Image Analysis. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2018. Springer International Publishing, pp. 440–448.

  > Lafarge,M.W., Bekkers,E.J., Pluim,J.P.W., Duits,R. and Veta,M. (2021) Roto-translation equivariant convolutional networks: Application to histopathology image analysis. Med. Image Anal., 68, 101849.

The package exports `RotGroupConv`, which is a convolution operator applied to
either a 4-dimensional input with shape `[width, height, nchannels, nbatches]`,
or applied to 5-dimensional input with `[width, height, nchannels, nbatches,
nrotations]`. The former the paper calls a "lifting layer" and the latter a
"group convolution layer".

Not really tested, use at your own risk.


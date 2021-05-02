def spectral_convolution(spectrum):
    spectrum_int = sorted([int(v) for v in spectrum.split()])
    convolution = []
    for i in spectrum_int:
        for j in spectrum_int:
            a = i-j
            if a>0:
                convolution.append(a)
    return convolution

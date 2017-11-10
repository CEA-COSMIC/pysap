##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import unittest
import os
import numpy

# Package import
import pisap
import pisap.extensions.transform
from pisap.data import get_sample_data
from pisap.base.transform import WaveletTransformBase


class TestAnalysisSynthesis(unittest.TestCase):
    """ Test the analysis/synthesis of an input image.
    """
    def setUp(self):
        """ Get the data from the server.
        """
        self.images = [
            get_sample_data(dataset_name="astro-fits"),
            get_sample_data(dataset_name="mri-slice-nifti")]
        print("[info] Image loaded for test: {0}.".format(
            [i.data.shape for i in self.images]))
        self.transforms = WaveletTransformBase.REGISTRY.values()
        print("[info] Found {0} transformations.".format(len(self.transforms)))
        self.nb_scales = [2, 3, 4]
        self.errors = [{
            2: {'BsplineWaveletTransformATrousAlgorithm': 0.0,
                'DecompositionOnScalingFunction': 0.16180419921875,
                'FeauveauWaveletTransformWithoutUndersampling': 0.0,
                'HaarWaveletTransform': 0.0,
                'HalfPyramidalTransform': 0.0,
                'IsotropicAndCompactSupportWaveletInFourierSpace':
                    0.226684570312,
                'LinearWaveletTransformATrousAlgorithm': 0.0,
                'MallatWaveletTransform79Filters': 0.0,
                'MeyerWaveletsCompactInFourierSpace': 0.224731445312,
                'MixedHalfPyramidalWTAndMedianMethod': 0.0,
                'MixedWTAndPMTMethod': 0.36480712890625,
                'MorphologicalMedianTransform': 0.0,
                'MorphologicalMinmaxTransform': 0.0,
                'MorphologicalPyramidalMinmaxTransform': 0.320068359375,
                'NonOrthogonalUndecimatedTransform': 0.0,
                'OnLine44AndOnColumn53': 0.0,
                'OnLine53AndOnColumn44': 0.0,
                'PyramidalBsplineWaveletTransform': 0.36566162109375,
                'PyramidalLaplacian': 0.0,
                'PyramidalLinearWaveletTransform': 0.3565673828125,
                'PyramidalMedianTransform': 0.32000732421875,
                'PyramidalWaveletTransformInFourierSpaceAlgo1':
                    0.3919677734375,
                'PyramidalWaveletTransformInFourierSpaceAlgo2':
                    0.39263916015625,
                'UndecimatedBiOrthogonalTransform': 0.0,
                'UndecimatedDiadicWaveletTransform': 0.0,
                'UndecimatedHaarTransformATrousAlgorithm': 0.0,
                'WaveletTransformInFourierSpace': 0.27435302734375,
                'WaveletTransformViaLiftingScheme': 0.0},
            3: {'BsplineWaveletTransformATrousAlgorithm': 0.0,
                'DecompositionOnScalingFunction': 0.30865478515625,
                'FeauveauWaveletTransformWithoutUndersampling': 0.0,
                'HaarWaveletTransform': 0.0,
                'HalfPyramidalTransform': 0.41143798828125,
                'IsotropicAndCompactSupportWaveletInFourierSpace': 1.0,
                'LinearWaveletTransformATrousAlgorithm': 0.0,
                'MallatWaveletTransform79Filters': 0.000244140625,
                'MeyerWaveletsCompactInFourierSpace': 1.0,
                'MixedHalfPyramidalWTAndMedianMethod': 0.40997314453125,
                'MixedWTAndPMTMethod': 0.69207763671875,
                'MorphologicalMedianTransform': 0.0,
                'MorphologicalMinmaxTransform': 0.0,
                'MorphologicalPyramidalMinmaxTransform': 0.40972900390625,
                'NonOrthogonalUndecimatedTransform': 0.0,
                'OnLine44AndOnColumn53': 0.0,
                'OnLine53AndOnColumn44': 0.0,
                'PyramidalBsplineWaveletTransform': 0.70208740234375,
                'PyramidalLaplacian': 0.0,
                'PyramidalLinearWaveletTransform': 0.69549560546875,
                'PyramidalMedianTransform': 0.633544921875,
                'PyramidalWaveletTransformInFourierSpaceAlgo1':
                    0.43280029296875,
                'PyramidalWaveletTransformInFourierSpaceAlgo2':
                    0.4273681640625,
                'UndecimatedBiOrthogonalTransform': 0.0,
                'UndecimatedDiadicWaveletTransform': 0.0,
                'UndecimatedHaarTransformATrousAlgorithm': 0.0,
                'WaveletTransformInFourierSpace': 0.28509521484375,
                'WaveletTransformViaLiftingScheme': 0.0},
            4: {'BsplineWaveletTransformATrousAlgorithm': 0.0,
                'DecompositionOnScalingFunction': 0.94378662109375,
                'FeauveauWaveletTransformWithoutUndersampling': 0.0,
                'HaarWaveletTransform': 0.0,
                'HalfPyramidalTransform': 0.78033447265625,
                'IsotropicAndCompactSupportWaveletInFourierSpace': 1.0,
                'LinearWaveletTransformATrousAlgorithm': 0.0,
                'MallatWaveletTransform79Filters': 0.00274658203125,
                'MeyerWaveletsCompactInFourierSpace': 1.0,
                'MixedHalfPyramidalWTAndMedianMethod': 0.76617431640625,
                'MixedWTAndPMTMethod': 0.9461669921875,
                'MorphologicalMedianTransform': 0.0,
                'MorphologicalMinmaxTransform': 0.0,
                'MorphologicalPyramidalMinmaxTransform': 0.60595703125,
                'NonOrthogonalUndecimatedTransform': 0.0,
                'OnLine44AndOnColumn53': 0.02081298828125,
                'OnLine53AndOnColumn44': 0.03411865234375,
                'PyramidalBsplineWaveletTransform': 0.9544677734375,
                'PyramidalLaplacian': 0.0,
                'PyramidalLinearWaveletTransform': 0.94207763671875,
                'PyramidalMedianTransform': 0.87689208984375,
                'PyramidalWaveletTransformInFourierSpaceAlgo1': 0.447265625,
                'PyramidalWaveletTransformInFourierSpaceAlgo2':
                    0.41900634765625,
                'UndecimatedBiOrthogonalTransform': 0.0,
                'UndecimatedDiadicWaveletTransform': 0.0098876953125,
                'UndecimatedHaarTransformATrousAlgorithm': 0.0,
                'WaveletTransformInFourierSpace': 0.27264404296875,
                'WaveletTransformViaLiftingScheme': 0.0}
        }, {
            2: {'BsplineWaveletTransformATrousAlgorithm': 0.0,
                'DecompositionOnScalingFunction': 0.00904083251953125,
                'FeauveauWaveletTransformWithoutUndersampling': 0.0,
                'HaarWaveletTransform': 0.0,
                'HalfPyramidalTransform': 0.0,
                'IsotropicAndCompactSupportWaveletInFourierSpace': 1.0,
                'LinearWaveletTransformATrousAlgorithm': 0.0,
                'MallatWaveletTransform79Filters': 0.057071685791015625,
                'MeyerWaveletsCompactInFourierSpace': 1.0,
                'MixedHalfPyramidalWTAndMedianMethod': 0.0,
                'MixedWTAndPMTMethod': 0.34236907958984375,
                'MorphologicalMedianTransform': 0.0,
                'MorphologicalMinmaxTransform': 0.0,
                'MorphologicalPyramidalMinmaxTransform': 0.11091995239257812,
                'NonOrthogonalUndecimatedTransform': 0.0,
                'OnLine44AndOnColumn53': 0.0,
                'OnLine53AndOnColumn44': 0.0,
                'PyramidalBsplineWaveletTransform': 0.34614944458007812,
                'PyramidalLaplacian': 0.0,
                'PyramidalLinearWaveletTransform': 0.33107757568359375,
                'PyramidalMedianTransform': 0.30455398559570312,
                'PyramidalWaveletTransformInFourierSpaceAlgo1':
                    0.52199935913085938,
                'PyramidalWaveletTransformInFourierSpaceAlgo2':
                    0.51966476440429688,
                'UndecimatedBiOrthogonalTransform': 0.05445098876953125,
                'UndecimatedDiadicWaveletTransform': 0.0,
                'UndecimatedHaarTransformATrousAlgorithm': 0.0,
                'WaveletTransformInFourierSpace': 0.5282745361328125,
                'WaveletTransformViaLiftingScheme': 0.0},
            3: {'BsplineWaveletTransformATrousAlgorithm': 0.0,
                'DecompositionOnScalingFunction': 0.039836883544921875,
                'FeauveauWaveletTransformWithoutUndersampling':
                    0.0041351318359375,
                'HaarWaveletTransform': 0.0,
                'HalfPyramidalTransform': 0.38523483276367188,
                'IsotropicAndCompactSupportWaveletInFourierSpace': 1.0,
                'LinearWaveletTransformATrousAlgorithm': 0.0,
                'MallatWaveletTransform79Filters': 0.093997955322265625,
                'MeyerWaveletsCompactInFourierSpace': 1.0,
                'MixedHalfPyramidalWTAndMedianMethod': 0.38259506225585938,
                'MixedWTAndPMTMethod': 0.58610153198242188,
                'MorphologicalMedianTransform': 0.0,
                'MorphologicalMinmaxTransform': 0.0,
                'MorphologicalPyramidalMinmaxTransform': 0.19281387329101562,
                'NonOrthogonalUndecimatedTransform': 0.031055450439453125,
                'OnLine44AndOnColumn53': 0.0,
                'OnLine53AndOnColumn44': 0.0,
                'PyramidalBsplineWaveletTransform': 0.59321975708007812,
                'PyramidalLaplacian': 0.0,
                'PyramidalLinearWaveletTransform': 0.57712936401367188,
                'PyramidalMedianTransform': 0.54430389404296875,
                'PyramidalWaveletTransformInFourierSpaceAlgo1':
                    0.51559829711914062,
                'PyramidalWaveletTransformInFourierSpaceAlgo2':
                    0.51426315307617188,
                'UndecimatedBiOrthogonalTransform': 0.087833404541015625,
                'UndecimatedDiadicWaveletTransform': 0.000240325927734375,
                'UndecimatedHaarTransformATrousAlgorithm': 0.0,
                'WaveletTransformInFourierSpace': 0.52921676635742188,
                'WaveletTransformViaLiftingScheme': 0.0},
            4: {'BsplineWaveletTransformATrousAlgorithm': 0.0,
                'DecompositionOnScalingFunction': 0.19489288330078125,
                'FeauveauWaveletTransformWithoutUndersampling':
                    0.03284454345703125,
                'HaarWaveletTransform': 0.0,
                'HalfPyramidalTransform': 0.63311386108398438,
                'IsotropicAndCompactSupportWaveletInFourierSpace': 1.0,
                'LinearWaveletTransformATrousAlgorithm': 0.0,
                'MallatWaveletTransform79Filters': 0.15792083740234375,
                'MeyerWaveletsCompactInFourierSpace': 1.0,
                'MixedHalfPyramidalWTAndMedianMethod': 0.62831497192382812,
                'MixedWTAndPMTMethod': 0.7991180419921875,
                'MorphologicalMedianTransform': 0.0,
                'MorphologicalMinmaxTransform': 0.0,
                'MorphologicalPyramidalMinmaxTransform': 0.23126983642578125,
                'NonOrthogonalUndecimatedTransform': 0.07872772216796875,
                'OnLine44AndOnColumn53': 0.00099945068359375,
                'OnLine53AndOnColumn44': 0.002185821533203125,
                'PyramidalBsplineWaveletTransform': 0.80150604248046875,
                'PyramidalLaplacian': 0.0,
                'PyramidalLinearWaveletTransform': 0.78143692016601562,
                'PyramidalMedianTransform': 0.74028396606445312,
                'PyramidalWaveletTransformInFourierSpaceAlgo1':
                    0.5211944580078125,
                'PyramidalWaveletTransformInFourierSpaceAlgo2':
                    0.51951217651367188,
                'UndecimatedBiOrthogonalTransform': 0.1413116455078125,
                'UndecimatedDiadicWaveletTransform': 0.05423736572265625,
                'UndecimatedHaarTransformATrousAlgorithm': 0.0,
                'WaveletTransformInFourierSpace': 0.52893447875976562,
                'WaveletTransformViaLiftingScheme': 0.0}
        }]

    def test_wavelet_transformations(self):
        """ Test all the registered transformations.
        """
        # from pprint import pprint
        # d = {}
        for errors, image in zip(self.errors, self.images):
            for nb_scale in self.nb_scales:
                # d[nb_scale] = {}
                for transform in self.transforms:
                    print("[info] Testing {0}...".format(transform))
                    transform = transform(nb_scale=nb_scale, verbose=1)
                    transform.data = image
                    transform.analysis()
                    # transform.show()
                    recim = transform.synthesis()
                    # recim.show()
                    mismatch = (1. - numpy.mean(
                        numpy.isclose(recim.data, image.data, atol=1e-8,
                                      rtol=1e-5)))
                    # d[nb_scale][transform.__class__.__name__] = mismatch
                    error = errors[nb_scale][transform.__class__.__name__]
                    self.assertTrue(mismatch < error + 1e-8)
            # pprint(d)

if __name__ == "__main__":
    unittest.main()

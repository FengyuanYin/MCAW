import unittest
import torch
from traceguard.adapters.wav2vec_compat import verify_checkpoint_parameters


class FakeModel:
    def __init__(self):
        self.values = {'encoder.conv.parametrizations.weight.original0': torch.zeros(2),
                       'encoder.conv.parametrizations.weight.original1': torch.zeros(2, 2),
                       'masked_spec_embed': torch.zeros(2)}
    def state_dict(self):
        return self.values
    def load_state_dict(self, values, strict):
        self.values.update(values)
    def eval(self):
        return self


class CompatibilityTests(unittest.TestCase):
    def test_legacy_parameters_loaded_exactly(self):
        model = FakeModel()
        source = {'wav2vec2.encoder.conv.weight_g': torch.tensor([1., 2.]),
                  'wav2vec2.encoder.conv.weight_v': torch.tensor([[3., 4.], [5., 6.]])}
        report = verify_checkpoint_parameters(model, source)
        self.assertEqual(report['verified_parameters'], 2)
        self.assertTrue(torch.equal(model.values['encoder.conv.parametrizations.weight.original0'], source['wav2vec2.encoder.conv.weight_g']))
    def test_encoder_missing_rejected(self):
        with self.assertRaisesRegex(ValueError, 'critical parameters missing'):
            verify_checkpoint_parameters(FakeModel(), {})
    def test_wrong_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            verify_checkpoint_parameters(FakeModel(), {'encoder.conv.weight_g': torch.zeros(3)})


if __name__ == '__main__':
    unittest.main()

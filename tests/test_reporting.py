import numpy as np
from picking.reporting import paired_interval


def test_paired_bootstrap_preserves_constant_effect_and_problem_clusters():
    values = np.full((3,4),0.25)
    result = paired_interval(values,['same','same','different','third'],42,samples=100)
    assert result['mean']==0.25
    assert np.allclose(result['ci95'],[0.25,0.25])

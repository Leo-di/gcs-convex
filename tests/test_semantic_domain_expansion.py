import itertools
import numpy as np
from src.semantic_domain_expansion import expand_domain


def test_adaptive_domain_keeps_endpoints_limits_and_semantic_margin_at_all_corners():
    a=np.zeros(10);b=np.r_[[.04,.01,.005],[.002]*7]
    limits=np.full(10,.2);answer=expand_domain(a,b,-limits,limits,.8,2.)
    lo,hi,record=answer;center=(a+b)/2
    assert np.all(lo<=np.minimum(a,b)) and np.all(hi>=np.maximum(a,b))
    assert np.all(lo>=-limits) and np.all(hi<=limits)
    # Exact affine separating-plane rule, with admissible translational and
    # rotational Lipschitz constants, independently evaluated at every corner.
    w=np.r_[[.6,.8],np.full(8,2.)]
    margins=[.8-w@(np.where(bits,hi,lo)-center)
             for bits in itertools.product([False,True],repeat=10)]
    assert min(margins)>=record['certified_lower_bound_m']-1e-10
    assert record['certified_lower_bound_m']>0


def test_low_slack_shrinks_domains_instead_of_disabling_rule_or_dropping_endpoint():
    a=np.zeros(10);b=np.r_[[.01,0,0],[0]*7];lo=np.full(10,-1.);hi=-lo
    tight=expand_domain(a,b,lo,hi,.03,2.)
    roomy=expand_domain(a,b,lo,hi,.5,2.)
    assert np.all(tight[1]-tight[0]<roomy[1]-roomy[0])
    assert expand_domain(a,b,lo,hi,.001,2.) is None

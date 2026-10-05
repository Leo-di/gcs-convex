import json
import numpy as np
from src.heldout_transfer_design import design
from src import screen_carry_regions as screening


def test_fixed_unseen_cohort_and_contact_coverage():
    d=design();assert d==design();assert len(d['scenes'])==16;assert len(d['cases'])==48
    contacts=[c for c in d['cases'] if c['contact']];assert len(contacts)==16
    assert {c['id'][-1] for c in contacts}=={'0','1','2'}
    assert not d['use_known_witnesses']
    for name,c in d['scenes'].items():
        walls={w['name']:w for w in c['world']['walls']};a=walls['shared_wall_upper'];b=walls['shared_wall_lower']
        assert np.isclose(a['center'][1]-a['size'][1]/2-b['center'][1]-b['size'][1]/2,d['changes'][name]['door_width'])


def test_no_witness_mode_never_reads_old_results(tmp_path,monkeypatch):
    def forbidden(*args):raise AssertionError('Old results must not be read')
    monkeypatch.setattr(screening,'known_witnesses',forbidden)
    class Oracle:
        calls=0
        def __init__(self,*args):pass
    monkeypatch.setattr(screening,'CarryOracle',Oracle)
    source=tmp_path/'raw/libraries/new';source.mkdir(parents=True)
    (tmp_path/'design.json').write_text(json.dumps({'scenes':{'new':{}}}))
    (source/'shared.json').write_text(json.dumps({'config':{},'candidates':{'x':{'rows':[{'phase':next(iter(screening.PICK_PHASES)),'base_q':[0,0,0],'arm_q':[0]*7}],'cover':{'regions':[]}}},'goals':{'x':{}}}))
    (source/'preparation.json').write_text(json.dumps({'connectivity':[]}))
    screening.screen_scene(tmp_path,'new',use_known_witnesses=False)
    report=json.loads((tmp_path/'screened/libraries/new/screen_report.json').read_text())
    assert report['use_known_witnesses'] is False
    assert report['candidates']['x']['known_points']==0

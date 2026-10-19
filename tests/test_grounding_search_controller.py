from src.grounding_search_controller import search_groundings


def test_bridge_repair_without_grasp_does_not_stop_search(tmp_path):
    called=[]
    def evaluate(candidate,remaining):
        called.append(candidate['id']);assert remaining>0
        good=candidate['id']=='b'
        return dict(process_exit=0,bridge={'accepted':True},operation={'accepted':good},physical={'accepted':good})
    result=search_groundings([{'id':x} for x in 'abc'],evaluate,10,tmp_path/'trace.json')
    assert result['selected']=='b' and called==['a','b']
    assert not result['attempts'][0]['complete']


def test_timed_out_execution_is_not_accepted(tmp_path):
    def evaluate(c,r):return dict(process_exit='timeout',bridge={'accepted':True},operation={'accepted':True},physical={'accepted':True})
    result=search_groundings([{'id':'a'}],evaluate,10,tmp_path/'trace.json')
    assert not result['success'] and result['selected'] is None

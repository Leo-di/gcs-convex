"""Publish the full-task online IRIS integration and its contact execution."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/results/online_iris_delivery_v1'))
    p.add_argument('--web',type=Path,default=Path('outputs/results/grasp_fix_web/planar/online_iris_delivery'))
    args=p.parse_args(); source=args.source; out=args.web; out.mkdir(parents=True,exist_ok=True)
    data={name:json.loads((source/file).read_text()) for name,file in
          [('protocol','protocol.json'),('attachment','online_attachment.json'),
           ('plan','result.json'),('offline','offline_result.json'),('physics','dynamics/result.json')]}
    data['reference']=json.loads((source/'reference/delivery_trace.json').read_text())
    for name,path in [('reference.html','reference/scene.html'),('physics.html','dynamics/scene.html'),
                      ('physics.json','dynamics/result.json'),('plan.json','result.json'),
                      ('attachment.json','online_attachment.json'),('protocol.json','protocol.json'),
                      ('regions.json','online_regions.json')]:
        shutil.copy2(source/path,out/name)
    (out/'data.json').write_text(json.dumps(data))
    template=Path('src/online_iris_delivery_view.html').read_text()
    (out/'index.html').write_text(template)
    link='<section id="online-delivery-link"><b>在线 IRIS 已接入完整任务的携物阶段</b><p><a href="../online_iris_delivery/">查看在线生成区域 → 语义 GCS → 完整抓放接触执行</a>。这是端点区域留出集成测试，不是抓放成功率基准。</p></section>'
    for page in [out.parent/'full_task_recheck/index.html',out.parent/'online_iris_100q/index.html',Path('src/online_iris_batch_view.html')]:
        text=page.read_text()
        if 'id="online-delivery-link"' not in text:
            page.write_text(text.replace('<section>',link+'\n<section>',1))
    print(out/'index.html')


if __name__=='__main__':main()

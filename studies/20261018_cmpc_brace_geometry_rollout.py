#!/usr/bin/env python3
"""Prerequisite ablation: original controller, CAD wrist collision in the plant.

The 8.64--23.52 mm wrist-corner discrepancy motivates replacing only the named
left_wrist_pad box by the existing wrist CAD convex collision mesh in memory.
This is a plant-model sensitivity test, not a corrected brace planner.
"""
import argparse,csv,importlib.util,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module(n,p):
 s=importlib.util.spec_from_file_location(n,ROOT/'studies'/p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
F=module('foot','20261016_cmpc_foot_motion.py');np,mj=F.np,F.mujoco
OUT=ROOT/'studies/runs/20261018-cmpc-brace-selection'
MEDIA=ROOT/'docs/lean/media/20261015-contact-audit'
BASE=ROOT/'studies/runs/20261016-cmpc-foot-motion/20261016-nominal'

def correct(m):
 g=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'left_wrist_pad');src=49
 assert m.geom_bodyid[g]==m.geom_bodyid[src] and m.geom_type[src]==mj.mjtGeom.mjGEOM_MESH
 fields=['geom_type','geom_dataid','geom_pos','geom_quat','geom_size','geom_rbound','geom_aabb']
 before={k:np.asarray(getattr(m,k)[g]).tolist() for k in fields}
 for k in fields:getattr(m,k)[g]=getattr(m,k)[src]
 return dict(geom_id=g,source_geom=src,fields=fields,before=before,
  after={k:np.asarray(getattr(m,k)[g]).tolist() for k in fields},
  collision='MuJoCo convex hull of existing wrist_yaw_link CAD mesh; CAD triangle witness audit remains required.')

class Recorder(F.Recorder):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.correction=correct(self.m);self.table=[];self.__class__.last=self
 def sample(self):
  super().sample()
  if len(self.rows)%10:return
  for i in range(self.d.ncon):
   c=self.d.contact[i]
   if 99 not in c.geom:continue
   first=c.geom2==99;g=int(c.geom1 if first else c.geom2);b=int(self.m.geom_bodyid[g])
   name=mj.mj_id2name(self.m,mj.mjtObj.mjOBJ_BODY,b)
   ff=np.zeros(6);mj.mj_contactForce(self.m,self.d,i,ff);out=c.frame[:3]*(1 if first else -1)
   pw=c.pos-.5*c.dist*out;R=self.d.xmat[b].reshape(3,3)
   self.table.append(dict(t=float(self.d.time),geom=g,body=name,normal_N=float(ff[0]),
    tangent_N=float(np.linalg.norm(ff[1:3])),point_body_m=(R.T@(pw-self.d.xpos[b])).tolist(),
    normal_body=(R.T@out).tolist(),gap_mm=float(c.dist*1000)))

def run():
 F.OUT=OUT;F.Recorder=Recorder
 F.run('cad_wrist')
 for p in OUT.glob('20261016-cad_wrist*'):p.rename(p.with_name(p.name.replace('20261016','20261018',1)))
 F.dump(OUT/'20261018-cad_wrist-table.json',Recorder.last.table)
 F.dump(OUT/'20261018-cad_wrist-correction.json',Recorder.last.correction)


def render():
 import imageio.v2 as imageio
 from PIL import Image,ImageDraw,ImageFont
 import height_video as hv
 records=[]
 for label,stem,corrected in [('Original wrist box',BASE,False),('CAD wrist collision',OUT/'20261018-cad_wrist',True)]:
  with Path(str(stem)+'.csv').open() as f:
   reader=csv.reader(l for l in f if not l.startswith('#'));head=next(reader);a=np.array(list(reader),float)
  ts=a[:,head.index('time')];qi=[i for i,h in enumerate(head) if h.startswith('qpos')]
  req,ix=hv.frame_samples(ts,30,1.)
  m,d=F.cs.load(ik_margin=0);change=correct(m) if corrected else None
  m.vis.global_.offwidth=960;m.vis.global_.offheight=540
  ren=mj.Renderer(m,540,960);cam=mj.MjvCamera();mj.mjv_defaultCamera(cam)
  cam.azimuth=140;cam.elevation=-12;cam.distance=3.4;cam.lookat[:]=[.55,-.1,.95]
  path=MEDIA/('20261018-brace-'+('cad-wrist' if corrected else 'original')+'.mp4')
  font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
  frames=[]
  with imageio.get_writer(path,fps=30,macro_block_size=1,ffmpeg_params=['-threads','1']) as writer:
   for j,k in enumerate(ix):
    d.qpos[:]=a[k,qi];mj.mj_forward(m,d);ren.update_scene(d,cam)
    im=Image.fromarray(ren.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,38),fill=(15,20,27))
    draw.text((12,6),f'{label} | Simulation t = {ts[k]:.2f} s | Playback 1x',font=font,fill='white');writer.append_data(np.asarray(im))
    if j in [0,15,72,120,150,180]:frames.append(im.resize((480,270)))
  ren.close();sheet=Image.new('RGB',(1440,540))
  for j,im in enumerate(frames):sheet.paste(im,((j%3)*480,(j//3)*270))
  sheet.save(path.with_suffix('.png'))
  rec=dict(label=label,source=str(stem)+'.csv',source_sha256=F.sha(str(stem)+'.csv'),video=str(path.relative_to(ROOT)),
   video_sha256=F.sha(path),frames=len(ix),fps=30,speed=1,simulation_end_s=float(ts[-1]),duration_s=len(ix)/30,
   sample_error_s=float(np.max(abs(ts[ix]-req))),model_correction=change)
  records.append(rec)
 F.dump(OUT/'20261018-videos.json',records)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','render']);a=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 run() if a.mode=='run' else render()

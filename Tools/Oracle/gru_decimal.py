"""Independent reset-after GRU forward and analytic adjoints, Decimal60."""
from decimal import Decimal as D, localcontext
from copy import deepcopy
import json
from pathlib import Path

def zeros(n,m):return [[D(0)]*m for _ in range(n)]
def sigmoid(x):return 1/(1+(-x).exp())
def tanh(x):return 2*sigmoid(2*x)-1

def fixture():
 x=[[[D(((b*11+t*5+i*3)%13)-6)/10 for i in range(2)] for t in range(3)] for b in range(2)]
 h=[[D('.15'),D('-.25')],[D('-.35'),D('.05')]]
 def weight(shift):return [[D(((i*7+j*3+shift)%11)-5)/10 for j in range(6)] for i in range(2)]
 ps=[weight(1),weight(4),[D(v)/10 for v in [1,-2,3,-1,2,-3]],[D(v)/10 for v in [-2,3,-1,2,-4,1]]]
 seed=[[[D(((b*7+t*3+j*5)%9)-4)/10 for j in range(2)] for t in range(3)] for b in range(2)]
 final=[[D('.3'),D('-.7')],[D('-.4'),D('.6')]]
 return x,h,ps,seed,final

def calculate(x,initial,ps,seed,final):
 wi,wh,bi,bh=ps;batch=len(x);hidden=2;time=3;input_features=2
 h=deepcopy(initial);sequence=[[] for _ in x];cache=[]
 for t in range(time):
  xa=[[sum(x[b][t][i]*wi[i][j] for i in range(input_features))+bi[j] for j in range(6)] for b in range(batch)]
  ha=[[sum(h[b][i]*wh[i][j] for i in range(hidden))+bh[j] for j in range(6)] for b in range(batch)]
  r=[[sigmoid(xa[b][j]+ha[b][j]) for j in range(hidden)] for b in range(batch)]
  z=[[sigmoid(xa[b][j+2]+ha[b][j+2]) for j in range(hidden)] for b in range(batch)]
  n=[[tanh(xa[b][j+4]+r[b][j]*ha[b][j+4]) for j in range(hidden)] for b in range(batch)]
  new=[[(1-z[b][j])*n[b][j]+z[b][j]*h[b][j] for j in range(hidden)] for b in range(batch)]
  cache.append((h,r,z,n,ha));h=new
  for b in range(batch):sequence[b].append(h[b][:])
 dx=[zeros(time,input_features) for _ in x];dwi=zeros(input_features,6);dwh=zeros(hidden,6);dbi=[D(0)]*6;dbh=[D(0)]*6;dh=deepcopy(final)
 for t in reversed(range(time)):
  old,r,z,n,ha=cache[t];da=zeros(batch,6);dc=zeros(batch,6);next_dh=zeros(batch,hidden)
  for b in range(batch):
   for j in range(hidden):
    g=dh[b][j]+seed[b][t][j];dn=g*(1-z[b][j])*(1-n[b][j]*n[b][j]);dz=g*(old[b][j]-n[b][j])*z[b][j]*(1-z[b][j]);dr=dn*ha[b][j+4]*r[b][j]*(1-r[b][j]);da[b][j]=dr;da[b][j+2]=dz;da[b][j+4]=dn;dc[b][j]=dr;dc[b][j+2]=dz;dc[b][j+4]=dn*r[b][j];next_dh[b][j]=g*z[b][j]
   for i in range(input_features):dx[b][t][i]=sum(da[b][j]*wi[i][j] for j in range(6))
   for i in range(hidden):next_dh[b][i]+=sum(dc[b][j]*wh[i][j] for j in range(6))
   for j in range(6):
    dbi[j]+=da[b][j];dbh[j]+=dc[b][j]
    for i in range(input_features):dwi[i][j]+=x[b][t][i]*da[b][j]
    for i in range(hidden):dwh[i][j]+=old[b][i]*dc[b][j]
  dh=next_dh
 return dict(state=h,sequence=sequence,input_gradient=dx,initial_gradient=dh,parameter_gradients=[dwi,dwh,dbi,dbh])

def flatten(v):
 if isinstance(v,list):return [x for child in v for x in flatten(child)]
 return [v]
def positions(v,p=()):
 if isinstance(v,list):
  for i,child in enumerate(v):yield from positions(child,p+(i,))
 else:yield p
def get(v,path):
 for i in path:v=v[i]
 return v
def put(v,path,value):
 for i in path[:-1]:v=v[i]
 v[path[-1]]=value

def calculate_report():
 with localcontext() as ctx:
  ctx.prec=60;x,h,ps,seed,final=fixture();result=calculate(x,h,ps,seed,final)
  def loss(args):
   r=calculate(*args,seed,final)
   return sum(a*b for a,b in zip(flatten(r['sequence']),flatten(seed)))+sum(a*b for a,b in zip(flatten(r['state']),flatten(final)))
  args=[x,h,ps];grads=[result['input_gradient'],result['initial_gradient'],result['parameter_gradients']];eps=D('1e-12');errors=[]
  for k,value in enumerate(args):
   for path in positions(value):
    plus=deepcopy(args);minus=deepcopy(args);put(plus[k],path,get(value,path)+eps);put(minus[k],path,get(value,path)-eps)
    numerical=(loss(plus)-loss(minus))/(2*eps);errors.append(abs(numerical-get(grads[k],path)))
  assert max(errors)<D('1e-20'),max(errors)
  report=dict(precision=60,input=x,initial_state=h,parameters=ps,sequence_seed=seed,final_seed=final,**result,finite_difference_components=len(errors),finite_difference_epsilon=eps,finite_difference_max_error=max(errors),convention='reset-after r,z,n; packed columns [input,3hidden], [hidden,3hidden]; separate input/recurrent biases')
  return report

def generate_sources():
 with localcontext() as ctx:
  ctx.prec=60;x,h,ps,seed,final=fixture();r=calculate(x,h,ps,seed,final)
 def literal(v):return '['+','.join(str(float(x)) for x in flatten(v))+']'
 code='// Generated independent Decimal60 GRU reset-after forward and VJP fixture.\nuse Tensor\nuse Tensor.Neural\nuse Tensor.Autograd\nuse GFX.GPU\nuse STD.Math\n\n'
 code+='func gru_oracle_near(actual:Tensor,expected:float[]) {\n    let data=actual.detach().cpu().values();assert(data.count()==expected.count())\n    var i=0;while i<data.count() {assert(Math.is_finite(data[i]) && Math.abs(data[i]-expected[i])<0.000002);i++}\n}\n'
 code+='func decimal_gru(device:GPU.Device?) {\n    var model=Neural.GRU("memory",2,2,seed:11)\n    var parameters=model.parameters();assert(parameters.count()==4)\n'
 for i,(name,v,shape) in enumerate(zip(['input_weight','recurrent_weight','input_bias','recurrent_bias'],ps,[[2,6],[2,6],[6],[6]])):
  code+=f'    assert(parameters[{i}].name()=="memory.{name}");var p{i}=parameters[{i}]\n    var data{i}:float[]={literal(v)};p{i}.replace(Tensor(data{i},{shape}))\n'
 for name,v,shape in [('xs',x,[2,3,2]),('hs',h,[2,2]),('ss',seed,[2,3,2]),('fs',final,[2,2])]:code+=f'    var {name}:float[]={literal(v)};var {name}_tensor=Tensor({name},{shape})\n'
 code+='    if let gpu=device {model.to(gpu);xs_tensor=xs_tensor.to(gpu);hs_tensor=hs_tensor.to(gpu);ss_tensor=ss_tensor.to(gpu);fs_tensor=fs_tensor.to(gpu)}\n    var input=Autograd.Variable(xs_tensor);var initial=Autograd.Variable(hs_tensor)\n    let result=model.unroll(input.value(),initial_state:initial.value(),return_sequence:true)\n    var inputs:Tensor[]=[input.value(),initial.value()];for parameter in parameters {inputs.append(parameter.value())}\n    var loss=result.state.multiply(fs_tensor).sum()\n    if let sequence=result.sequence {\n        gru_oracle_near(sequence,'+literal(r['sequence'])+')\n        loss=loss.add(sequence.multiply(ss_tensor).sum())\n    } else {panic("expected GRU sequence")}\n    var gradients=Autograd.grad(loss,inputs)\n'
 code+='    gru_oracle_near(result.state,'+literal(r['state'])+')\n'
 for i,v in enumerate([r['input_gradient'],r['initial_gradient']]+r['parameter_gradients']):code+=f'    gru_oracle_near(gradients[{i}],{literal(v)})\n'
 code+='}\ntest "GRU reset-after matches Decimal sequence state and six VJPs on CPU" {decimal_gru(null)}\ntest "GRU reset-after matches Decimal on Metal" {\n    if !GPU.Device.is_supported() {return}\n    var device=GPU.Device(GPU.DeviceSettings(debug:true));decimal_gru(device)\n}\n'
 return code,json.dumps(calculate_report(),default=str,indent=2)+"\n"

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
 root=Path(__file__).resolve().parents[2];code,report=generate_sources()
 for path,content in {root/'Tests/Consumer/Tests/GRUOracle.sx':code,root/'Tools/Oracle/GRU_REPORT.json':report}.items():
  if args.check:
   if not path.exists() or path.read_text()!=content:raise SystemExit('Fixture mismatch: '+str(path))
  else:path.write_text(content)
 print('GRU Decimal fixtures verified' if args.check else 'GRU Decimal fixtures generated')
if __name__=='__main__':main()

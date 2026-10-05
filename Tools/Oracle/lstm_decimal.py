"""Independent Decimal60 LSTM gate equations and analytic reverse derivatives."""
from decimal import Decimal as D,localcontext
from copy import deepcopy
from pathlib import Path
import json

def zeros(n,m):return [[D(0)]*m for _ in range(n)]
def sigmoid(x):return 1/(1+(-x).exp())
def tanh(x):return 2*sigmoid(2*x)-1

def fixture():
 x=[[[D(((b*11+t*5+j*3)%13)-6)/10 for j in range(2)] for t in range(3)] for b in range(2)]
 h=[[D('.15'),D('-.25')],[D('-.35'),D('.05')]];c=[[D('-.2'),D('.3')],[D('.1'),D('-.4')]]
 def w(shift):return [[D(((i*7+j*3+shift)%11)-5)/10 for j in range(8)] for i in range(2)]
 ps=[w(1),w(4),[D(v)/10 for v in [1,-2,3,-1,2,-3,4,-2]],[D(v)/10 for v in [-2,3,-1,2,-4,1,-3,2]]]
 seed=[[[D(((b*7+t*3+j*5)%9)-4)/10 for j in range(2)] for t in range(3)] for b in range(2)]
 sh=[[D('.3'),D('-.7')],[D('-.4'),D('.6')]];sc=[[D('-.2'),D('.5')],[D('.7'),D('-.1')]]
 return x,h,c,ps,seed,sh,sc

def calculate(x,initial_h,initial_c,ps,seed,sh,sc):
 wi,wh,bi,bh=ps;h=deepcopy(initial_h);c=deepcopy(initial_c);seq=[[] for _ in x];cache=[]
 for t in range(3):
  a=[[sum(x[b][t][j]*wi[j][k]+h[b][j]*wh[j][k] for j in range(2))+bi[k]+bh[k] for k in range(8)] for b in range(2)]
  i=[[sigmoid(row[j]) for j in range(2)] for row in a];f=[[sigmoid(row[j+2]) for j in range(2)] for row in a];g=[[tanh(row[j+4]) for j in range(2)] for row in a];o=[[sigmoid(row[j+6]) for j in range(2)] for row in a]
  cn=[[f[b][j]*c[b][j]+i[b][j]*g[b][j] for j in range(2)] for b in range(2)];hn=[[o[b][j]*tanh(cn[b][j]) for j in range(2)] for b in range(2)]
  cache.append((h,c,i,f,g,o,cn));h,c=hn,cn
  for b in range(2):seq[b].append(h[b][:])
 dx=[zeros(3,2) for _ in x];dw=zeros(2,8);du=zeros(2,8);db=[D(0)]*8;dh=deepcopy(sh);dc=deepcopy(sc)
 for t in reversed(range(3)):
  old_h,old_c,i,f,g,o,new_c=cache[t];da=zeros(2,8);next_dc=zeros(2,2)
  for b in range(2):
   for j in range(2):
    gh=dh[b][j]+seed[b][t][j];tc=tanh(new_c[b][j]);gc=dc[b][j]+gh*o[b][j]*(1-tc*tc)
    da[b][j]=gc*g[b][j]*i[b][j]*(1-i[b][j]);da[b][j+2]=gc*old_c[b][j]*f[b][j]*(1-f[b][j]);da[b][j+4]=gc*i[b][j]*(1-g[b][j]*g[b][j]);da[b][j+6]=gh*tc*o[b][j]*(1-o[b][j]);next_dc[b][j]=gc*f[b][j]
   for j in range(2):dx[b][t][j]=sum(da[b][k]*wi[j][k] for k in range(8))
   for k in range(8):
    db[k]+=da[b][k]
    for j in range(2):dw[j][k]+=x[b][t][j]*da[b][k];du[j][k]+=old_h[b][j]*da[b][k]
  dh=[[sum(da[b][k]*wh[j][k] for k in range(8)) for j in range(2)] for b in range(2)];dc=next_dc
 return dict(hidden=h,cell=c,sequence=seq,input_gradient=dx,initial_hidden_gradient=dh,initial_cell_gradient=dc,parameter_gradients=[dw,du,db,db[:]])

def flatten(v):
 if isinstance(v,list):return [x for child in v for x in flatten(child)]
 return [v]
def positions(v,path=()):
 if isinstance(v,list):
  for i,child in enumerate(v):yield from positions(child,path+(i,))
 else:yield path
def get(v,path):
 for i in path:v=v[i]
 return v
def put(v,path,value):
 for i in path[:-1]:v=v[i]
 v[path[-1]]=value

def calculate_report():
 with localcontext() as ctx:
  ctx.prec=60;x,h,c,ps,seed,sh,sc=fixture();r=calculate(x,h,c,ps,seed,sh,sc);args=[x,h,c,ps];grads=[r['input_gradient'],r['initial_hidden_gradient'],r['initial_cell_gradient'],r['parameter_gradients']];eps=D('1e-12');errors=[]
  def loss(v):
   result=calculate(*v,seed,sh,sc)
   return sum(a*b for a,b in zip(flatten(result['sequence']),flatten(seed)))+sum(a*b for a,b in zip(flatten(result['hidden']),flatten(sh)))+sum(a*b for a,b in zip(flatten(result['cell']),flatten(sc)))
  for k,value in enumerate(args):
   for path in positions(value):
    plus=deepcopy(args);minus=deepcopy(args);put(plus[k],path,get(value,path)+eps);put(minus[k],path,get(value,path)-eps);errors.append(abs((loss(plus)-loss(minus))/(2*eps)-get(grads[k],path)))
  assert max(errors)<D('1e-20'),max(errors)
  report=dict(precision=60,input=x,initial_hidden=h,initial_cell=c,parameters=ps,sequence_seed=seed,hidden_seed=sh,cell_seed=sc,**r,finite_difference_components=len(errors),finite_difference_epsilon=eps,finite_difference_max_error=max(errors),convention='i,f,g,o; separate row-major [input,4hidden]/[hidden,4hidden] weights and two biases')
  return report

def generate_sources():
 with localcontext() as ctx:
  ctx.prec=60;x,h,c,ps,seed,sh,sc=fixture();r=calculate(x,h,c,ps,seed,sh,sc)
 def literal(v):return '['+','.join(str(float(a)) for a in flatten(v))+']'
 code='// Generated independent Decimal60 LSTM forward and seven VJP fixture.\nuse Tensor\nuse Tensor.Neural\nuse Tensor.Autograd\nuse GFX.GPU\nuse STD.Math\n\n'
 code+='func lstm_oracle_near(actual:Tensor,expected:float[]) {\n    let data=actual.detach().cpu().values();assert(data.count()==expected.count())\n    var i=0;while i<data.count() {assert(Math.is_finite(data[i]) && Math.abs(data[i]-expected[i])<0.000002);i++}\n}\n'
 code+='func decimal_lstm(device:GPU.Device?) {\n    var model=Neural.LSTM("memory",2,2,seed:11)\n    var parameters=model.parameters();assert(parameters.count()==4)\n'
 for i,(name,v,shape) in enumerate(zip(['input_weight','recurrent_weight','input_bias','recurrent_bias'],ps,[[2,8],[2,8],[8],[8]])):
  code+=f'    assert(parameters[{i}].name()=="memory.{name}");var p{i}=parameters[{i}]\n    var data{i}:float[]={literal(v)};p{i}.replace(Tensor(data{i},{shape}))\n'
 for name,v,shape in [('xs',x,[2,3,2]),('hs',h,[2,2]),('cs',c,[2,2]),('ss',seed,[2,3,2]),('sh',sh,[2,2]),('sc',sc,[2,2])]:code+=f'    var {name}:float[]={literal(v)};var {name}_tensor=Tensor({name},{shape})\n'
 code+='    if let gpu=device {model.to(gpu);xs_tensor=xs_tensor.to(gpu);hs_tensor=hs_tensor.to(gpu);cs_tensor=cs_tensor.to(gpu);ss_tensor=ss_tensor.to(gpu);sh_tensor=sh_tensor.to(gpu);sc_tensor=sc_tensor.to(gpu)}\n    var input=Autograd.Variable(xs_tensor);var initial_h=Autograd.Variable(hs_tensor);var initial_c=Autograd.Variable(cs_tensor)\n    let result=model.unroll(input.value(),initial_state:Neural.LSTMState(hidden:initial_h.value(),cell:initial_c.value()),return_sequence:true)\n    var inputs:Tensor[]=[input.value(),initial_h.value(),initial_c.value()];for parameter in parameters {inputs.append(parameter.value())}\n    var loss=result.state.hidden.multiply(sh_tensor).sum().add(result.state.cell.multiply(sc_tensor).sum())\n    if let sequence=result.sequence {\n        lstm_oracle_near(sequence,'+literal(r['sequence'])+')\n        loss=loss.add(sequence.multiply(ss_tensor).sum())\n    } else {panic("expected LSTM sequence")}\n    var gradients=Autograd.grad(loss,inputs)\n'
 code+='    lstm_oracle_near(result.state.hidden,'+literal(r['hidden'])+')\n    lstm_oracle_near(result.state.cell,'+literal(r['cell'])+')\n'
 for i,v in enumerate([r['input_gradient'],r['initial_hidden_gradient'],r['initial_cell_gradient']]+r['parameter_gradients']):code+=f'    lstm_oracle_near(gradients[{i}],{literal(v)})\n'
 code+='}\ntest "LSTM matches Decimal sequence hidden cell and seven VJPs on CPU" {decimal_lstm(null)}\ntest "LSTM matches Decimal on Metal" {\n    if !GPU.Device.is_supported() {return}\n    var device=GPU.Device(GPU.DeviceSettings(debug:true));decimal_lstm(device)\n}\n'
 return code,json.dumps(calculate_report(),default=str,indent=2)+'\n'

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
 root=Path(__file__).resolve().parents[2];code,report=generate_sources()
 for path,content in {root/'Tests/Consumer/Tests/LSTMOracle.sx':code,root/'Tools/Oracle/LSTM_REPORT.json':report}.items():
  if args.check:
   if not path.exists() or path.read_text()!=content:raise SystemExit('Fixture mismatch: '+str(path))
  else:path.write_text(content)
 print('LSTM Decimal fixtures verified' if args.check else 'LSTM Decimal fixtures generated')
if __name__=='__main__':main()

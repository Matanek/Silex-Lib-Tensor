"""Decimal reference Transformer forward and analytic reverse-mode equations."""
from decimal import Decimal as D, localcontext
import json

def zeros(n,m):return [[D(0)]*m for _ in range(n)]
def add(a,b):return [[x+y for x,y in zip(u,v)] for u,v in zip(a,b)]
def matrix(rows):return [[D(str(v)) for v in row] for row in rows]
def linear(x,w,b):return [[sum(v*w[i][j] for i,v in enumerate(row))+b[j] for j in range(len(b))] for row in x]
def linear_vjp(x,w,g):
 dx=[[sum(v*w[i][j] for j,v in enumerate(row)) for i in range(len(w))] for row in g]
 dw=[[sum(row[i]*up[j] for row,up in zip(x,g)) for j in range(len(w[0]))] for i in range(len(w))]
 db=[sum(row[j] for row in g) for j in range(len(w[0]))]
 return dx,dw,db

def norm(x,weight,bias,epsilon):
 normalized=[];inverse=[]
 for row in x:
  mean=sum(row)/len(row);variance=sum((v-mean)**2 for v in row)/len(row)
  inv=1/(variance+epsilon).sqrt();inverse.append(inv);normalized.append([(v-mean)*inv for v in row])
 output=[[v*w+b for v,w,b in zip(row,weight,bias)] for row in normalized]
 def vjp(g):
  dx=[]
  for z,inv,row in zip(normalized,inverse,g):
   dz=[v*w for v,w in zip(row,weight)];mean=sum(dz)/len(dz);corr=sum(a*b for a,b in zip(z,dz))/len(dz)
   dx.append([inv*(v-mean-a*corr) for a,v in zip(z,dz)])
  return dx,[sum(row[j]*z[j] for row,z in zip(g,normalized)) for j in range(len(weight))],[sum(row[j] for row in g) for j in range(len(weight))]
 return output,vjp

def attention(x,parameters,mask):
 wq,bq,wk,bk,wv,bv,wo,bo=parameters
 q=linear(x,wq,bq);k=linear(x,wk,bk);v=linear(x,wv,bv)
 factor=1/D(2).sqrt();mixed=zeros(len(x),4);probabilities=[]
 present=[any(any(head[i]) for head in mask) for i in range(len(x))]
 for head in range(2):
  probs=[];cols=range(head*2,head*2+2)
  for i in range(len(x)):
   exps=[(sum(q[i][c]*k[j][c] for c in cols)*factor).exp() if mask[head][i][j] else D(0) for j in range(len(x))]
   total=sum(exps);p=[e/total if total else D(0) for e in exps];probs.append(p)
   for c in cols:mixed[i][c]=sum(p[j]*v[j][c] for j in range(len(x)))
  probabilities.append(probs)
 output=linear(mixed,wo,bo);output=[row if present[i] else [D(0)]*4 for i,row in enumerate(output)]
 def vjp(g):
  upstream=[row if present[i] else [D(0)]*4 for i,row in enumerate(g)]
  dm,dwo,dbo=linear_vjp(mixed,wo,upstream);dq=zeros(len(x),4);dk=zeros(len(x),4);dv=zeros(len(x),4)
  for head in range(2):
   cols=range(head*2,head*2+2)
   for i in range(len(x)):
    p=probabilities[head][i];dp=[sum(dm[i][c]*v[j][c] for c in cols) for j in range(len(x))]
    correction=sum(a*b for a,b in zip(p,dp));ds=[a*(b-correction)*factor for a,b in zip(p,dp)]
    for c in cols:
     dq[i][c]=sum(ds[j]*k[j][c] for j in range(len(x)))
     for j in range(len(x)):dk[j][c]+=ds[j]*q[i][c];dv[j][c]+=p[j]*dm[i][c]
  gradients=[linear_vjp(x,w,z) for w,z in zip([wq,wk,wv],[dq,dk,dv])]
  return add(add(gradients[0][0],gradients[1][0]),gradients[2][0]),[item for _,dw,db in gradients for item in [dw,db]]+[dwo,dbo]
 return output,probabilities,vjp

def feed_forward(x,parameters):
 wi,bi,wo,bo=parameters;pre=linear(x,wi,bi);hidden=[[max(D(0),v) for v in row] for row in pre];output=linear(hidden,wo,bo)
 def vjp(g):
  dh,dwo,dbo=linear_vjp(hidden,wo,g);dpre=[[v if activation>0 else D(0) for v,activation in zip(row,z)] for row,z in zip(dh,pre)]
  dx,dwi,dbi=linear_vjp(x,wi,dpre);return dx,[dwi,dbi,dwo,dbo]
 return output,vjp,pre

def fixture():
 x=matrix([[.125,-.5,.75,.25],[.5,1.25,-.75,.125],[.25,-.5,1.5,-.25]])
 seed=matrix([[.2,-.3,.7,-1.1],[-.4,.8,.3,.1],[.6,-.2,-.5,.9]])
 def weight(rows,cols,shift):return [[D(((i*7+j*3+shift)%13)-6)/D(10) for j in range(cols)] for i in range(rows)]
 def bias(n,shift):return [D(((i*3+shift)%7)-3)/D(10) for i in range(n)]
 ps=[item for n in range(4) for item in [weight(4,4,n+1),bias(4,n+1)]]
 ps += [[D('1.2'),D('.8'),D('1.1'),D('.9')],[D('.1'),D('-.2'),D('.05'),D('.15')],
        [D('.9'),D('1.3'),D('.7'),D('1.1')],[D('-.05'),D('.1'),D('.2'),D('-.15')],
        weight(4,6,5),bias(6,4),weight(6,4,8),bias(4,2)]
 mask=[[[False,False,False],[True,True,False],[True,False,True]],[[False,False,False],[False,True,False],[True,True,True]]]
 return x,seed,ps,mask

def calculate(norm_first):
 x,seed,ps,mask=fixture();eps=D('.001')
 if norm_first:
  n1,n1v=norm(x,ps[8],ps[9],eps);a,weights,av=attention(n1,ps[:8],mask);r=add(x,a)
  n2,n2v=norm(r,ps[10],ps[11],eps);f,fv,pre=feed_forward(n2,ps[12:]);out=add(r,f)
  df,pgf=fv(seed);dn2,pgn2w,pgn2b=n2v(df);dr=add(seed,dn2);da,pga=av(dr);dn1,pgn1w,pgn1b=n1v(da);dx=add(dr,dn1)
 else:
  a,weights,av=attention(x,ps[:8],mask);r,n1v=norm(add(x,a),ps[8],ps[9],eps)
  f,fv,pre=feed_forward(r,ps[12:]);out,n2v=norm(add(r,f),ps[10],ps[11],eps)
  d2,pgn2w,pgn2b=n2v(seed);df,pgf=fv(d2);d1,pgn1w,pgn1b=n1v(add(d2,df));da,pga=av(d1);dx=add(d1,da)
 return dict(norm_first=norm_first,input=x,seed=seed,parameters=ps,mask=mask,epsilon=eps,output=out,weights=weights,input_gradient=dx,parameter_gradients=pga+[pgn1w,pgn1b,pgn2w,pgn2b]+pgf,relu_min_distance=min(abs(v) for row in pre for v in row))


def flatten(value):
 if isinstance(value,list):return [v for row in value for v in flatten(row)]
 return [value]
def literal(value):return '['+','.join(str(float(v)) for v in flatten(value))+']'
def generate_sources():
 with localcontext() as ctx:
  ctx.prec=60;cases=[calculate(True),calculate(False)]
 names=['block.attention.'+part+'.'+slot for part in ['query','key','value','output'] for slot in ['weight','bias']]
 names += ['block.'+part+'.'+slot for part in ['attention_norm','feed_forward_norm','feed_forward.input','feed_forward.output'] for slot in ['weight','bias']]
 shapes=[[4,4],[4]]*4+[[4],[4]]*2+[[4,6],[6],[6,4],[4]]
 code='// Generated Decimal Transformer forward/VJP fixture.\nuse Tensor\nuse Tensor.Neural\nuse Tensor.Autograd\nuse GFX.GPU\nuse STD.Math\n\n'
 code+='func transformer_oracle_near(actual:Tensor,expected:float[]) {\n    let data=actual.detach().cpu().values();assert(data.count()==expected.count())\n    var i=0;while i<data.count() {assert(Math.is_finite(data[i]) && Math.abs(data[i]-expected[i])<0.000008);i++}\n}\n'
 code+='func decimal_transformer(device:GPU.Device?,norm_first:bool) {\n    var model=Neural.TransformerBlock("block",4,2,6,seed:19,norm_first:norm_first,epsilon:0.001)\n    var parameters=model.parameters();assert(parameters.count()==16)\n'
 for i,(name,value,shape) in enumerate(zip(names,cases[0]['parameters'],shapes)):
  code+=f'    assert(parameters[{i}].name()=="{name}");var p{i}=parameters[{i}]\n    var data{i}:float[]={literal(value)};p{i}.replace(Tensor(data{i},{shape}))\n'
 code+=f"    var xs:float[]={literal(cases[0]['input'])};var seeds:float[]={literal(cases[0]['seed'])}\n"
 flags=flatten(cases[0]['mask']);code+='    var flags:bool[]=['+','.join(str(v).lower() for v in flags)+']\n'
 code+='    var x=Tensor(xs,[1,3,4]);var seed=Tensor(seeds,[1,3,4]);var mask=Tensor(flags,[1,2,3,3])\n    if let gpu=device {model.to(gpu);x=x.to(gpu);seed=seed.to(gpu);mask=mask.to(gpu)}\n    var input=Autograd.Variable(x);var inputs:Tensor[]=[input.value()]\n    for parameter in parameters {inputs.append(parameter.value())}\n    let result=model.encode(input.value(),mask:mask,return_weights:true)\n    var gradients=Autograd.grad(result.output,inputs,seed:seed)\n'
 for index,case in enumerate(cases):
  code+='    '+('if norm_first {' if index==0 else 'else {')+'\n'
  code+='        transformer_oracle_near(result.output,'+literal(case['output'])+')\n'
  code+='        if let weights=result.weights {transformer_oracle_near(weights,'+literal(case['weights'])+')} else {panic("expected Transformer attention weights")}\n'
  for i,g in enumerate([case['input_gradient']]+case['parameter_gradients']):code+=f'        transformer_oracle_near(gradients[{i}],{literal(g)})\n'
  code+='    }\n'
 code+='}\ntest "Transformer pre and post norm match Decimal output and sixteen parameter VJPs on CPU" {decimal_transformer(null,true);decimal_transformer(null,false)}\ntest "Transformer pre and post norm match Decimal on Metal" {\n    if !GPU.Device.is_supported() {return}\n    var device=GPU.Device(GPU.DeviceSettings(debug:true));decimal_transformer(device,true);decimal_transformer(device,false)\n}\n'
 report=dict(precision=60,parameter_names=names,parameter_shapes=shapes,cases=cases,conversion='Separate row-major affine matrices [input,output], feature vectors for norm and bias; no packing or transpose',absolute_tolerance=0.000008)
 return code,json.dumps(report,default=str,indent=2)+'\n'


def main():
 from pathlib import Path
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
 root=Path(__file__).resolve().parents[2];code,report=generate_sources()
 outputs={root/'Tests/Consumer/Tests/TransformerOracle.sx':code,root/'Tools/Oracle/TRANSFORMER_REPORT.json':report}
 for path,content in outputs.items():
  if args.check:
   if not path.exists() or path.read_text()!=content:raise SystemExit('Fixture mismatch: '+str(path))
  else:path.write_text(content)
 print('Transformer Decimal fixtures verified' if args.check else 'Transformer Decimal fixtures generated')

if __name__=='__main__':main()

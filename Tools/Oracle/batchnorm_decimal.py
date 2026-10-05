"""Independent closed-form BatchNorm oracle; no Tensor or external framework."""
from decimal import Decimal, localcontext
from pathlib import Path
import argparse, json
parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true");args=parser.parse_args()
cases=[("dense",[3,2],["1","2","4","-1","7","5"],[".2","-.3",".7",".4","-.5",".8"],[[0,2,4],[1,3,5]]),
       ("nchw",[2,2,1,2],["1","3","-2","4","5","9","6","8"],[".2","-.3",".7",".4","-.5",".8",".1","-.6"],[[0,1,4,5],[2,3,6,7]])]
rows=[]
with localcontext() as context:
    context.prec=50
    gamma=list(map(Decimal,["1.5","-.75"]));beta=list(map(Decimal,[".25",".5"]))
    for name,shape,values,seeds,groups in cases:
        x=list(map(Decimal,values));g=list(map(Decimal,seeds));y=[Decimal(0)]*len(x);dx=y.copy()
        means=[];variances=[];dg=[];db=[];running_mean=[];running_variance=[]
        for c,indices in enumerate(groups):
            n=Decimal(len(indices));mean=sum(x[i] for i in indices)/n
            variance=sum((x[i]-mean)**2 for i in indices)/n
            inverse=1/(variance+Decimal(".01")).sqrt()
            h={i:(x[i]-mean)*inverse for i in indices}
            sg=sum(g[i] for i in indices);sgh=sum(g[i]*h[i] for i in indices)
            for i in indices:
                y[i]=gamma[c]*h[i]+beta[c]
                dx[i]=gamma[c]*inverse*(g[i]-sg/n-h[i]*sgh/n)
            means.append(mean);variances.append(variance);dg.append(sgh);db.append(sg)
            running_mean.append(Decimal(".25")*mean)
            running_variance.append(Decimal(".75")+Decimal(".25")*variance*n/(n-1))
        rows.append(dict(name=name,shape=shape,input=values,seed=seeds,output=list(map(str,y)),
                         input_gradient=list(map(str,dx)),weight_gradient=list(map(str,dg)),bias_gradient=list(map(str,db)),
                         mean=list(map(str,means)),population_variance=list(map(str,variances)),
                         running_mean=list(map(str,running_mean)),running_variance=list(map(str,running_variance))))
report=dict(oracle="Decimal 50 digits; closed-form channel derivative, independent of composite autograd",
            epsilon=".01",momentum=".25",weight=["1.5","-.75"],bias=[".25",".5"],cases=rows)
path=Path(__file__).with_name("BATCHNORM_REPORT.json");content=json.dumps(report,indent=2)+"\n"
if args.check:
    if path.read_text()!=content:raise SystemExit("BatchNorm report differs from the independent Decimal oracle")
else:path.write_text(content)

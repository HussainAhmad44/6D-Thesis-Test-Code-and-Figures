#!/usr/bin/env python3
"""Reproducible thesis tests for the user's ACTUAL fractional 6D map and saved S-box.
Run: python 6D_Thesis_Test_Suite.py --out thesis_test_figures
Dependencies: numpy matplotlib. Uses matplotlib Agg for screenshot-ready PNGs.
Important: numerical diagnostics do not prove hyperchaos or cipher security.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES="XYZWUV"
INITIAL=np.array([.11,.23,.37,.41,.59,.73],dtype=float)
DEFAULT=dict.fromkeys("abcde",.08)

def step(s, params):
    """Cyclic 6D fractional map. s shape (...,6); each coordinate uses rolled neighbors."""
    out=np.empty_like(s,dtype=float)
    a,b,c,d,e=[params[k] for k in "abcde"]
    for i in range(6):
        x,y,z,w,u,v=[s[..., (i+j)%6] for j in range(6)]
        out[...,i]=np.mod(2*x+a*np.sin(2*np.pi*y)/(2*np.pi)
            +b*np.cos(2*np.pi*z)/(2*np.pi)+c*np.log1p(w)/2
            +d*np.arctan(u)+e*np.tan(np.arctan(v)/2),1.)
    return out

def saveplot(path):
    plt.tight_layout();plt.savefig(path,dpi=220,bbox_inches="tight");plt.close()

def bifurcation(out, parameter, variable, lo,hi, samples, transient, keep):
    grid=np.linspace(lo,hi,samples)
    states=np.tile(INITIAL,(samples,1))
    # vectorize by parameter via explicit row-grouping, retaining same map definition
    for t in range(transient+keep):
        a,b,c,d,e=[np.full(samples,DEFAULT[k]) for k in "abcde"]
        coeff=dict(a=a,b=b,c=c,d=d,e=e)
        coeff[parameter]=grid
        nxt=np.empty_like(states)
        for i in range(6):
            x,y,z,w,u,v=[states[:,(i+j)%6] for j in range(6)]
            nxt[:,i]=np.mod(2*x+coeff["a"]*np.sin(2*np.pi*y)/(2*np.pi)
                +coeff["b"]*np.cos(2*np.pi*z)/(2*np.pi)+coeff["c"]*np.log1p(w)/2
                +coeff["d"]*np.arctan(u)+coeff["e"]*np.tan(np.arctan(v)/2),1.)
        states=nxt
        if t==transient: observed=np.empty((keep,samples))
        if t>=transient: observed[t-transient,:]=states[:,NAMES.index(variable)]
    plt.figure(figsize=(10,5.5))
    plt.plot(np.broadcast_to(grid,observed.shape).ravel(),observed.ravel(),",",color="black",alpha=.42)
    plt.title(f"Bifurcation: {variable} versus parameter {parameter} (fractional 6D map)")
    plt.xlabel(f"{parameter} (other parameters fixed at 0.08)")
    plt.ylabel(variable);plt.ylim(0,1)
    saveplot(out/f"01_bifurcation_{variable}_vs_{parameter}.png")

def trajectory(out,steps=4000,burn=1000):
    s=INITIAL.copy()
    data=[]
    for i in range(burn+steps):
        s=step(s,DEFAULT)
        if i>=burn:data.append(s.copy())
    data=np.asarray(data)
    plt.figure(figsize=(11,5))
    for j in range(3):plt.plot(data[:400,j],lw=.8,label=NAMES[j])
    plt.legend();plt.xlabel("Post-transient iteration");plt.ylabel("State value")
    plt.title("6D map: X, Y, Z trajectories")
    saveplot(out/"02_trajectories.png")
    plt.figure(figsize=(7,6))
    plt.scatter(data[:,0],data[:,1],s=1,alpha=.25)
    plt.xlabel("X");plt.ylabel("Y");plt.title("Phase portrait: X versus Y")
    saveplot(out/"03_phase_portrait.png")
    x=data[:,0]
    plt.figure(figsize=(8,5));plt.hist(x,bins=40,edgecolor="white")
    plt.xlabel("X");plt.ylabel("Count");plt.title("Histogram of X (not a uniformity proof)")
    saveplot(out/"04_histogram_X.png")
    lags=np.arange(1,41)
    corr=[float(np.corrcoef(x[:-k],x[k:])[0,1]) for k in lags]
    plt.figure(figsize=(9,5));plt.bar(lags,corr);plt.axhline(0,color="black",lw=.8)
    plt.xlabel("Lag");plt.ylabel("Pearson autocorrelation");plt.title("X lag autocorrelation")
    saveplot(out/"05_autocorrelation_X.png")
    return {"X_mean":float(x.mean()),"X_std":float(x.std()),
            "X_lag1_corr":corr[0],"X_lag5_corr":corr[4]}

def largest_lyapunov(params,steps=4000,burn=1000,epsilon=1e-8):
    """Finite-difference tangent estimate; modulo discontinuities may bias it.
    Report as numerical indicator only, not a rigorous Lyapunov spectrum."""
    x=INITIAL.copy()
    for _ in range(burn):x=step(x,params)
    v=np.array([1.,2.,3.,4.,5.,6.]);v/=np.linalg.norm(v)
    acc=0.
    for _ in range(steps):
        y=step(x,params)
        # wrapped difference on 6-torus, avoids artificial distance near modulo boundary
        y2=step(np.mod(x+epsilon*v,1),params)
        delta=(y2-y+.5)%1-.5
        norm=np.linalg.norm(delta)/epsilon
        if not np.isfinite(norm) or norm<=0:return float("nan")
        acc+=np.log(norm)
        v=delta/np.linalg.norm(delta)
        x=y
    return float(acc/steps)

def lyapunov_scan(out,parameter,lo,hi,count):
    grid=np.linspace(lo,hi,count)
    vals=[]
    for p in grid:
        params=DEFAULT.copy();params[parameter]=float(p)
        vals.append(largest_lyapunov(params,steps=1400,burn=500))
    plt.figure(figsize=(9,5));plt.plot(grid,vals,lw=1.2)
    plt.axhline(0,color="black",ls="--",lw=1)
    plt.xlabel(parameter);plt.ylabel("Estimated largest exponent / iteration")
    plt.title("Finite-difference Lyapunov indicator (exploratory)")
    saveplot(out/f"06_lyapunov_vs_{parameter}.png")
    return {"parameter":parameter,"at_default":largest_lyapunov(DEFAULT,steps=3000,burn=1000),
            "scan_min":float(np.nanmin(vals)),"scan_max":float(np.nanmax(vals))}

def fwht(a):
    a=np.asarray(a,dtype=np.int32).copy()
    h=1
    while h<256:
        z=a.reshape(-1,2*h)
        l=z[:,:h].copy();r=z[:,h:].copy()
        z[:,:h]=l+r;z[:,h:]=l-r
        h*=2
    return a

def sbox_tests(out,path):
    s=np.loadtxt(path,delimiter=",",dtype=np.int32).reshape(-1)
    if len(s)!=256 or min(s)<0 or max(s)>255 or len(set(s))!=256:
        raise ValueError("Expected a bijective 256-element 8-bit S-box")
    xs=np.arange(256,dtype=np.int32)
    masks=np.arange(1,256)
    parity=lambda a: np.array([int(v).bit_count()&1 for v in a],dtype=np.int32)
    nl=[]
    for mask in masks:
        walsh=fwht(1-2*parity(s&mask))
        nl.append(128-int(np.max(np.abs(walsh)))//2)
    nl=np.array(nl)
    ddt=np.zeros((256,256),dtype=np.int32)
    for dx in range(256):
        dy=s^s[xs^dx]
        ddt[dx]=np.bincount(dy,minlength=256)
    du=int(ddt[1:].max())
    sac=np.zeros((8,8),dtype=float)
    for inputbit in range(8):
        diff=s^s[xs^(1<<inputbit)]
        for outputbit in range(8):
            sac[inputbit,outputbit]=np.mean((diff>>outputbit)&1)
    # BIC-SAC: for each pair of output bit flips, test independence on each input bit
    bic_errors=[]
    for i in range(8):
        diff=s^s[xs^(1<<i)]
        bits=((diff[:,None]>>np.arange(8))&1)
        for j in range(8):
            for k in range(j+1,8):
                joint=np.mean(bits[:,j]*bits[:,k])
                bic_errors.append(abs(joint-np.mean(bits[:,j])*np.mean(bits[:,k])))
    plt.figure(figsize=(9,4.5));plt.hist(nl,bins=np.arange(nl.min()-.5,nl.max()+1.5))
    plt.xlabel("Component Boolean function nonlinearity");plt.ylabel("Number of masks")
    plt.title("S-box nonlinearity distribution (255 nonzero output masks)")
    saveplot(out/"07_sbox_nonlinearity.png")
    plt.figure(figsize=(7,6));plt.imshow(sac,vmin=0,vmax=1,cmap="viridis")
    plt.colorbar(label="Flip probability")
    plt.xticks(range(8));plt.yticks(range(8))
    plt.xlabel("Output bit");plt.ylabel("Input bit")
    plt.title("S-box SAC matrix (ideal value: 0.5)")
    saveplot(out/"08_sbox_SAC_heatmap.png")
    plt.figure(figsize=(8,6));plt.imshow(ddt[1:],aspect="auto",cmap="viridis")
    plt.colorbar(label="Number of solutions")
    plt.xlabel("Output XOR difference");plt.ylabel("Input XOR difference (1–255)")
    plt.title("S-box differential distribution table")
    saveplot(out/"09_sbox_DDT.png")
    with open(out/"sbox_SAC_matrix.csv","w",newline="") as f:csv.writer(f).writerows(sac)
    with open(out/"sbox_DDT_matrix.csv","w",newline="") as f:csv.writer(f).writerows(ddt)
    return {"bijective":True,"nonlinearity_min":int(nl.min()),"nonlinearity_mean":float(nl.mean()),
        "differential_uniformity":du,"max_differential_probability":du/256,
        "SAC_mean":float(sac.mean()),"SAC_min":float(sac.min()),"SAC_max":float(sac.max()),
        "BIC_SAC_mean_independence_error":float(np.mean(bic_errors)),
        "BIC_note":"This is a pairwise SAC-independence diagnostic, NOT BIC-NL."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="thesis_test_figures")
    ap.add_argument("--sbox",default=str(Path(__file__).with_name("6d_xor_annealed_sbox.csv")))
    ap.add_argument("--parameter",choices=list("abcde"),default="a")
    ap.add_argument("--variable",choices=list(NAMES),default="X")
    ap.add_argument("--min",type=float,default=0.0)
    ap.add_argument("--max",type=float,default=0.3)
    ap.add_argument("--samples",type=int,default=400)
    ap.add_argument("--transient",type=int,default=700)
    ap.add_argument("--keep",type=int,default=100)
    ap.add_argument("--lyapunov-samples",type=int,default=25)
    args=ap.parse_args()
    if args.min>=args.max:ap.error("--min must be less than --max")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    bifurcation(out,args.parameter,args.variable,args.min,args.max,args.samples,args.transient,args.keep)
    report={"model":"cyclic fractional 6D map with a=b=c=d=e=0.08",
            "initial_state":INITIAL.tolist(),
            "bifurcation":{"parameter":args.parameter,"variable":args.variable,
             "min":args.min,"max":args.max,"samples":args.samples,
             "transient":args.transient,"keep":args.keep},
            "trajectory":trajectory(out),
            "lyapunov_indicator":lyapunov_scan(out,args.parameter,args.min,args.max,args.lyapunov_samples),
            "sbox":sbox_tests(out,args.sbox)}
    (out/"test_results.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print("Saved PNG figures and CSVs to:",out.resolve())
if __name__=="__main__":main()

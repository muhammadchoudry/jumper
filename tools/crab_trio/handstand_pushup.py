import sys; sys.path.insert(0,"/root/jumper/tools/crab_trio")
import math, numpy as np, mujoco, itertools, handstand as H
m,d=H.setup(); dt=m.opt.timestep
E0,W0,J0=0.5,0.1,0.6
iJ3=[H.JI["LF_J3_joint"],H.JI["RF_J3_joint"]]; sg=[1.0,-1.0]
def run(amp, splay, kp, kd=0.0, setp=88.0, reps=3, period=2.4, record=None):
    base=H.vec(H.claws(elbow=E0,wrist=W0,jaw=J0)); low=H.vec(H.claws(elbow=E0+amp,wrist=W0,jaw=J0,yaw=90-splay))
    H.place_handstand(base,pitch_deg=90); rec=[]; zs=[]; nds=[]; fail=None
    T=0.8+reps*period+0.8
    for k in range(int(T/dt)):
        t=k*dt
        a=0 if t<0.8 or t>0.8+reps*period else 0.5-0.5*math.cos(2*math.pi*(t-0.8)/period)
        tgt=(base+a*(low-base)).copy(); nd=H.nose_down_deg(d); c=kp*math.radians(nd-setp)+kd*d.qvel[4]
        for i,s in zip(iJ3,sg): tgt[i]+=s*c
        H.step(tgt,rec); mujoco.mj_comPos(m,d)
        if t>0.6: zs.append(d.subtree_com[H.BASE][2]); nds.append(nd)
        nc,other=H.floor_contacts(d)
        if t>0.3 and fail is None and (other or nc==0 or nd<60): fail=round(t,2)
        if record is not None: record(t,d)
    rec=np.array(rec); zs=np.array(zs)
    return fail, zs.max()-zs.min(), min(nds), max(nds), rec.max(), 100*(rec>0.98*H.PEAK).mean()
if __name__=="__main__":
    res=[]
    for amp,splay,kp,kd in itertools.product((0.9,1.3,1.7),(0,10,20),(2,3,4.5),(0.0,0.1)):
        f,tr,nmin,nmax,tm,sat=run(amp,splay,kp,kd)
        res.append((f is None and sat<0.5, tr, amp,splay,kp,kd,nmin,nmax,tm,sat,f))
    res.sort(key=lambda r:(r[0],r[1]),reverse=True)
    for r in res[:8]: print("held %s travel %.3f m | elbow %.1f splay %d kp %.1f kd %.2f | nose-down %.0f..%.0f | peak %.2f maxed %.1f%% | fell %s"%r)

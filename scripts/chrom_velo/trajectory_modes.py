"""Small, descriptive six-stage accessibility models with explicit missingness.

Weighted LOOCV compares a constant, line, and two one-knot continuous lines.
Fits use the supplied linear normalized accessibility, never trajectory z-scores.
Count-derived measurement variances provide conditional sampling diagnostics;
they are not biological replicate uncertainty and ignore model-selection error.
"""
from __future__ import annotations
import numpy as np
from trajectory_classification import classify_supported_modes

MODE_LABELS = {0:'insufficient',1:'stable/no_supported_change',2:'increasing',
               3:'decreasing',4:'transient_peak',5:'transient_trough',6:'uncertain_complex'}
MODEL_LABELS = {0:'constant',1:'linear',2:'hinge_14hpf',3:'hinge_16hpf'}


def fit_trajectories(values, times, observed=None, weights=None, measurement_variance=None,
                     complexity_penalty=.05, min_observations=4, rate_floor=.05,
                     rate_z=3., lack_of_fit_threshold=4.):
    """Fit arrays shaped (..., T), returning fitted values and per-hour rates.

    Weights are positive relative precision weights (e.g. library coverage).
    The candidate score is weighted leave-one-stage-out MSE plus
    complexity_penalty*(n_parameters-1)*weighted observed variance. Knot choices
    are fixed in advance, not fitted to make UMAP directions attractive.

    Missing observations remain NaN in predictions and both adjacent rates;
    no absent stage is filled or bridged. Prediction clipping to zero is applied
    to full and leave-one-out predictions. A poor conditional residual fit is
    labelled uncertain; rate support additionally requires |rate| above both a
    practical floor and rate_z times the conditional sampling SE. When no
    measurement variances are supplied, rate SE is unavailable and the practical
    floor alone is used. "Stable" means no supported change under this procedure.
    """
    y=np.asarray(values,dtype=float); times=np.asarray(times,dtype=float)
    if y.ndim<1 or y.shape[-1]!=len(times) or len(times)<4:
        raise ValueError('Need matching trajectories and at least four timepoints')
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Times must be finite and strictly increasing')
    if any(not np.isfinite(x) or x<0 for x in [complexity_penalty,rate_floor,rate_z]):
        raise ValueError('Penalties, rate floor, and rate multiplier must be nonnegative')
    if min_observations<4 or min_observations>len(times):
        raise ValueError('min_observations must lie between four and T')
    if np.any(y[np.isfinite(y)]<0): raise ValueError('Accessibility cannot be negative')
    shape=y.shape[:-1]; T=len(times); yy=y.reshape(-1,T); n=len(yy)
    obs=np.isfinite(y) if observed is None else np.broadcast_to(np.asarray(observed,bool),y.shape)&np.isfinite(y)
    ww=np.ones(y.shape) if weights is None else np.broadcast_to(np.asarray(weights,float),y.shape)
    obs=obs&np.isfinite(ww)&(ww>0)
    variance=None
    if measurement_variance is not None:
        variance=np.broadcast_to(np.asarray(measurement_variance,float),y.shape)
        if np.any(variance[np.isfinite(variance)]<0): raise ValueError('Variance cannot be negative')
        obs &= np.isfinite(variance)
        variance=variance.reshape(-1,T)
    oo=obs.reshape(-1,T); ww=np.where(oo,ww.reshape(-1,T),0.)
    ww=np.divide(ww,ww.sum(axis=1,keepdims=True),out=np.zeros_like(ww),where=ww.sum(axis=1,keepdims=True)>0)
    out=dict(fitted=np.full((n,T),np.nan),rates=np.full((n,T-1),np.nan),
             rate_se=np.full((n,T-1),np.nan),rate_supported=np.zeros((n,T-1),bool),
             mode=np.zeros(n,np.int8),model=np.full(n,-1,np.int8),fit_valid=oo.sum(axis=1)>=min_observations,
             cv_error=np.full(n,np.nan),cv_improvement=np.full(n,np.nan),
             lack_of_fit=np.full(n,np.nan),observed_count=oo.sum(axis=1).astype(np.int8))
    if n==0: return {k:v.reshape(shape+v.shape[1:]) for k,v in out.items()}
    q=(times-times[0])/(times[-1]-times[0])
    designs=[np.ones((T,1)),np.column_stack([np.ones(T),q])]
    # The declared biological times are fixed knots; invalid domains omit them.
    for knot in [14.,16.]:
        z=(knot-times[0])/(times[-1]-times[0])
        designs.append(np.column_stack([np.ones(T),q,np.maximum(q-z,0.)]))
    group_weights,groups=np.unique(ww,axis=0,return_inverse=True)
    delta=np.diff(times)
    D=np.zeros((T-1,T))
    for t in range(T-1): D[t,t]=-1/delta[t];D[t,t+1]=1/delta[t]
    for group,w in enumerate(group_weights):
        ids=np.flatnonzero((groups==group)&out['fit_valid']); present=w>0
        if not len(ids): continue
        vals=np.where(present[None,:],yy[ids],0.)
        center=np.sum(vals*w,axis=1)
        scale=np.sum(w*(vals-center[:,None])**2,axis=1)
        errors=[];predictions=[];operators=[];scores=[]
        for model,B in enumerate(designs):
            valid_model=np.linalg.matrix_rank(B[present])==B.shape[1]
            if model>=2:
                knot=[14.,16.][model-2]
                valid_model &= (np.sum(present&(times<=knot))>=2 and np.sum(present&(times>=knot))>=2)
            H=B@np.linalg.pinv(B.T@(w[:,None]*B),rcond=1e-10)@(B.T*w)
            leverage=np.diag(H)
            valid_model &= np.all(leverage[present]<1.-1e-6)
            pred=np.maximum(vals@H.T,0.)
            if valid_model:
                # Exact linear-model LOO before nonnegative prediction clipping.
                linear_pred=vals@H.T
                loo=vals-np.divide(vals-linear_pred,1.-leverage,out=np.zeros_like(vals),where=(1.-leverage)>1e-6)
                loo=np.maximum(loo,0.)
                error=np.sum(w*(vals-loo)**2,axis=1)
            else: error=np.full(len(ids),np.inf)
            errors.append(error);predictions.append(pred);operators.append(H)
            scores.append(error+complexity_penalty*(B.shape[1]-1)*scale)
        chosen=np.argmin(np.asarray(scores),axis=0)
        best_error=np.asarray(errors)[chosen,np.arange(len(ids))]
        out['cv_error'][ids]=best_error
        base=errors[0]
        out['cv_improvement'][ids]=np.divide(base-best_error,base,out=np.zeros(len(ids)),where=base>1e-14)
        out['model'][ids]=chosen
        for model,H in enumerate(operators):
            use=chosen==model; take=ids[use]
            if not len(take): continue
            pred=predictions[model][use]
            rates=np.diff(pred,axis=1)/delta
            endpoints=present[:-1]&present[1:]
            rates[:,~endpoints]=np.nan
            fitted=pred.copy();fitted[:,~present]=np.nan
            out['fitted'][take]=fitted;out['rates'][take]=rates
            conditional_ok=np.ones(len(take),bool)
            if variance is not None:
                var=np.where(present[None,:],variance[take],0.)
                # Clipping changes the derivative operator near zero. Keep the
                # larger of linear and active-set delta variances; this remains
                # a conditional approximation, not full selection uncertainty.
                op=D@H
                linear_variance=var@(op*op).T
                stage_variance=var@(H*H).T
                adjacent_covariance=var@(H[:-1]*H[1:]).T
                unclipped=np.where(present[None,:],yy[take],0.)@H.T
                active=(unclipped>0).astype(float)
                clipped_variance=(stage_variance[:,:-1]*active[:,:-1]+stage_variance[:,1:]*active[:,1:]
                    -2*adjacent_covariance*active[:,:-1]*active[:,1:])/(delta*delta)
                se=np.sqrt(np.maximum(np.maximum(linear_variance,clipped_variance),0.))
                se[:,~endpoints]=np.nan;out['rate_se'][take]=se
                residual=np.where(present[None,:],yy[take]-pred,0.)
                noise=np.sum(w*var,axis=1)
                observed_error=np.sum(w*residual**2,axis=1)
                lof=np.divide(observed_error,noise,out=np.zeros(len(take)),where=noise>0)
                lof[(noise==0)&(observed_error>1e-12)]=np.inf
                out['lack_of_fit'][take]=lof
                conditional_ok=lof<=lack_of_fit_threshold
                threshold=np.maximum(rate_floor,rate_z*se)
            else:
                threshold=rate_floor
            support=np.isfinite(rates)&(np.abs(rates)>threshold)&conditional_ok[:,None]
            out['rate_supported'][take]=support
            out['mode'][take]=classify_supported_modes(rates,support,quality_valid=conditional_ok,rate_floor=rate_floor)
    return {k:v.reshape(shape+v.shape[1:]) for k,v in out.items()}

"""Phase M Step 1: fetch GOES X-ray flares + F10.7 for recovered MagNet span (1998-2019).
- GOES flares: NCEI yearly goes-xrs-report_YYYY.txt (1998-2017; 2018-19 = deep solar min, treated as no-flare).
- F10.7: CDAS OMNI2 hourly F10_INDEX, resampled daily.
Caches _goes_flares.pkl, _f107.pkl. Reuses CDAS fetch pattern from _symh_probe.py.
"""
import os, re, time, numpy as np, pandas as pd, requests

# ---------- GOES flares ----------
FCACHE="_goes_flares.pkl"
FBASE="https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs"
# class letter -> W/m^2 scale; peak flux = scale * (NN/10)
SCALE={'A':1e-8,'B':1e-7,'C':1e-6,'M':1e-5,'X':1e-4}
LINE=re.compile(r'^\d{5}(\d{2})(\d{2})(\d{2})\s+(\d{4}).*?([ABCMX])\s+(\d{2})\s+G(?:OES|\d{2})')

def yr4(yy):
    yy=int(yy); return 2000+yy if yy<70 else 1900+yy

if os.path.exists(FCACHE):
    flares=pd.read_pickle(FCACHE)
else:
    rows=[]
    for y in range(1998,2018):
        fn=f"goes-xrs-report_{y}-ytd.txt" if y==2017 else f"goes-xrs-report_{y}.txt"
        url=f"{FBASE}/{fn}"
        print(' fetch',fn,flush=True)
        for k in range(4):
            try:
                r=requests.get(url,timeout=120); r.raise_for_status(); break
            except Exception as e:
                if k==3: raise
                time.sleep(3*(k+1))
        n0=len(rows)
        for ln in r.text.splitlines():
            m=LINE.match(ln)
            if not m: continue
            yy,mm,dd,hhmm,cls,nn=m.groups()
            try:
                t=pd.Timestamp(f"{yr4(yy):04d}-{int(mm):02d}-{int(dd):02d} {hhmm[:2]}:{hhmm[2:]}")
            except Exception: continue
            flux=SCALE[cls]*(int(nn)/10.0)
            rows.append((t,cls,flux))
        print('   parsed',len(rows)-n0,flush=True)
    flares=pd.DataFrame(rows,columns=['t','cls','flux']).sort_values('t').reset_index(drop=True)
    flares.to_pickle(FCACHE)
print(f'flares: {len(flares)}  M+ {(flares.flux>=1e-5).sum()}  X {(flares.flux>=1e-4).sum()}  '
      f'span {flares.t.min()} .. {flares.t.max()}')

# ---------- F10.7 (LISIRD Penticton, adjusted to 1 AU) ----------
F7CACHE="_f107.pkl"
LISIRD="https://lasp.colorado.edu/lisird/latis/dap/penticton_radio_flux.json?time%3E=1998-01-01&time%3C=2020-01-01"

if os.path.exists(F7CACHE):
    f107=pd.read_pickle(F7CACHE)
else:
    print(' fetch F10.7 (LISIRD)',flush=True)
    for k in range(4):
        try:
            r=requests.get(LISIRD,timeout=300); r.raise_for_status(); break
        except Exception as e:
            if k==3: raise
            time.sleep(3*(k+1))
    s=r.json()['penticton_radio_flux']['samples']
    jd=np.array([x['time'] for x in s])
    val=np.array([x['adjusted_flux'] for x in s],dtype=float)
    t=pd.to_datetime((jd-2440587.5)*86400.0,unit='s')   # JD -> UTC
    f107=pd.Series(val,index=t)
    f107[f107>=9999]=np.nan
    f107=f107.resample('1D').mean().rename('f107')
    f107.to_pickle(F7CACHE)
print(f'f107 daily: {len(f107)}  NaN {f107.isna().mean():.3f}  range {f107.min():.0f}-{f107.max():.0f}')
print('done')

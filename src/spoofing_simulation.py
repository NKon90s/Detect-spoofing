
import pandas as pd

################################################################################
#
#  With the following code different spoofign scenarios can be simulated such as:
#   - Pseudorange changes due to spoofing
#   - Slow, incremental pseudorange changes due to spoofing
#   - Doppler changes due to spoofing
#   - SNR changes
#   - Replay delays
#
#################################################################################
 
# constants
L1_LAMBDA = 0.190293672798365   # L1 wavelength in meters
C = 299792458.0                 # Speed of light in m/s

def add_common_offset(df, start_time, end_time, offset_m, sats=None):

    """Add same pseudorange offset (meters) to specified sats (None = all)."""

    df2 = df.copy()
    mask_time = (df2['time'] >= pd.to_datetime(start_time)) & (df2['time'] <= pd.to_datetime(end_time))
    if sats is not None:
        mask_sat = df2['sv'].isin(sats)
        mask = mask_time & mask_sat
    else:
        mask = mask_time
    df2.loc[mask, 'pseudorange'] = df2.loc[mask, 'pseudorange'] + offset_m
    df2.loc[mask, 'attack_label'] = 1
    df2.loc[mask, 'attack_type'] = 'common offset'
    df2.loc[~df2.index.isin(df2[mask].index), 'attack_label'] = df2.loc[~df2.index.isin(df2[mask].index), 'attack_label'].fillna(0)
    df2.loc[~df2.index.isin(df2[mask].index), 'attack_type'] = df2.loc[~df2.index.isin(df2[mask].index), 'attack_type'].fillna(0)
    return df2


def add_ramp_offset(df, start_time, end_time, start_offset, end_offset, sats=None):

    """Linearly ramp pseudorange offset from start_offset to end_offset over interval."""

    df2 = df.copy()
    mask_time = (df2['time'] >= pd.to_datetime(start_time)) & (df2['time'] <= pd.to_datetime(end_time))
    times = df2.loc[mask_time, 'time'].astype('int64')//10**9

    if len(times)==0:
        return df2
    
    t0 = times.min()
    t1 = times.max()
    duration = t1 - t0 if t1!=t0 else 1.0
    # compute per-row offset
    frac = (times - t0) / duration
    offsets = start_offset + frac * (end_offset - start_offset)
    idxs = df2.loc[mask_time].index

    if sats is not None:
        idxs = df2.loc[mask_time & df2['sv'].isin(sats)].index
        times = df2.loc[idxs, 'time'].astype('int64')//10**9
        t0 = times.min(); t1 = times.max(); duration = max(1.0, t1-t0)
        frac = (times - t0) / duration
        offsets = start_offset + frac * (end_offset - start_offset)

    df2.loc[idxs, 'pseudorange'] = df2.loc[idxs, 'pseudorange'] + offsets.values
    df2.loc[idxs, 'attack_label'] = 1
    df2.loc[idxs, 'attack_type'] = 'ramp offset'
    df2.loc[~df2.index.isin(idxs), 'attack_label'] = df2.loc[~df2.index.isin(idxs), 'attack_label'].fillna(0)
    df2.loc[~df2.index.isin(idxs), 'attack_type'] = df2.loc[~df2.index.isin(idxs), 'attack_type'].fillna(0)
    return df2

def inject_doppler_offset(df, start_time, end_time, doppler_delta_hz, sats=None):

    """Add doppler offset (Hz). If you want to break consistency, leave pr unchanged."""

    df2 = df.copy()
    mask_time = (df2['time'] >= pd.to_datetime(start_time)) & (df2['time'] <= pd.to_datetime(end_time))
    if sats is not None:
        mask = mask_time & df2['sv'].isin(sats)
    else:
        mask = mask_time
    df2.loc[mask, 'doppler'] = df2.loc[mask, 'doppler'] + doppler_delta_hz
    df2.loc[mask, 'attack_label'] = 1
    df2.loc[mask, 'attack_type'] = 'doppler offset'
    df2.loc[~df2.index.isin(df2[mask].index), 'attack_label'] = df2.loc[~df2.index.isin(df2[mask].index), 'attack_label'].fillna(0)
    df2.loc[~df2.index.isin(df2[mask].index), 'attack_type'] = df2.loc[~df2.index.isin(df2[mask].index), 'attack_type'].fillna(0)
    return df2


def insert_cycle_slip(df, start_time, end_time, slip_cycles=1, sats=None):

    """Introduce an abrupt integer cycle slip into phase (add slip_cycles * lambda)."""

    df2 = df.copy()
    mask_time = (df2['time'] >= pd.to_datetime(start_time)) & (df2['time'] <= pd.to_datetime(end_time))
    if sats is not None:
        mask = mask_time & df2['sv'].isin(sats)
    else:
        mask = mask_time
    df2.loc[mask, 'phase'] = df2.loc[mask, 'phase'] + slip_cycles * L1_LAMBDA
    df2.loc[mask, 'attack_label'] = 1
    df2.loc[mask, 'attack_type'] = 'cycle slip'
    df2.loc[~df2.index.isin(df2[mask].index), 'attack_label'] = df2.loc[~df2.index.isin(df2[mask].index), 'attack_label'].fillna(0)
    df2.loc[~df2.index.isin(df2[mask].index), 'attack_type'] = df2.loc[~df2.index.isin(df2[mask].index), 'attack_type'].fillna(0)
    return df2


def change_snr(df, start_time, end_time, snr_delta_db, sats=None):

    """Increase/decrease SNR by snr_delta_db for the interval."""

    df2 = df.copy()
    mask_time = (df2['time'] >= pd.to_datetime(start_time)) & (df2['time'] <= pd.to_datetime(end_time))
    if sats is not None:
        mask = mask_time & df2['sv'].isin(sats)
    else:
        mask = mask_time
    df2.loc[mask, 'snr'] = df2.loc[mask, 'snr'] + snr_delta_db
    df2.loc[mask, 'attack_label'] = 1
    df2.loc[mask, 'attack_type'] = 'SNR change'
    df2.loc[~df2.index.isin(df2[mask].index), 'attack_label'] = df2.loc[~df2.index.isin(df2[mask].index), 'attack_label'].fillna(0)
    df2.loc[~df2.index.isin(df2[mask].index), 'attack_type'] = df2.loc[~df2.index.isin(df2[mask].index), 'attack_type'].fillna(0)
    return df2


def add_replay_delay(df, start_time, end_time, delay_seconds, sats=None):
    """
    Simulate a replay by *increasing pseudorange by c * delay_seconds*.
    c = speed of light (m/s)
    """
    delay_m = C * delay_seconds
    return add_common_offset(df, start_time, end_time, delay_m, sats=sats)







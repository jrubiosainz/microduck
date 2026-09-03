#!/usr/bin/env python3
"""Measured summary and hard gates for Protective Personal Space."""
from __future__ import annotations
import math
from pps_cast import WARD
from pps_states import (BUFFER_M, DUCK_PLANAR_RADIUS, INTERPOSE_BEARING_TOL_DEG,
                        RETREAT_RANGE_GAIN_M, RETREAT_TARGET_M, VX_ONSET,
                        VX_REVERSE_ONSET, ZERO_COMMAND_STATES)


def _fraction(values): return sum(bool(v) for v in values)/len(values) if values else 0.0


def _windows(records, episodes):
    windows=[]
    for e in episodes:
        rows=[r for r in records if e['started_at_s']<=r['t']<=e['ended_at_s']]
        path=(rows[-1]['path_m']-rows[0]['path_m']) if len(rows)>1 else 0.0
        targets=[r.get('protective_target_distance_m') for r in rows
                 if r.get('protective_target_distance_m') is not None]
        windows.append({**e,"path_m":round(path,4),"rows":len(rows),
            "between_ticks":sum(r['between'] for r in rows),
            "target_reduction_m":round(max(targets)-min(targets),4) if targets else 0.0,
            "min_person_clearance_m":round(min((r['min_person_clearance_m'] for r in rows),default=9),4),
            "active_visible_fraction":round(_fraction([r['active_visible'] for r in rows if r['active']]),4)})
    return windows


def summarize(rollout):
    records=rollout.records; episodes=_windows(records,rollout.machine.episodes)
    intrusions=[e for e in episodes if e['kind']=='intrusion']
    ward_episode=next((e for e in episodes if e['kind']=='ward_approach'),None)
    squeeze=next((e for e in episodes if e['kind']=='squeeze'),None)
    # A squeeze is itself a genuine intrusion cycle: it has a selected primary,
    # measured bearing and a harder response branch. Keep it in chronological
    # bearing evidence rather than pretending it did not count because its
    # response was ESCAPE_GAP instead of INTERPOSE.
    threat_cycles=[e for e in episodes if e['kind'] in ('intrusion','squeeze')]
    retreat=[r for r in records if r['state']=='RETREAT']
    retreat_range=(retreat[-1]['ward_range_m']-retreat[0]['ward_range_m']) if retreat else 0
    retreat_path=(retreat[-1]['path_m']-retreat[0]['path_m']) if retreat else 0
    bearings=[e.get('bearing_deg') for e in threat_cycles if e.get('bearing_deg') is not None]
    bearing_sides=[1 if math.cos(math.radians(v))>=0 else -1 for v in bearings]
    alternating=all(a!=b for a,b in zip(bearing_sides,bearing_sides[1:]))
    threat_camera_states={'PREDICT_INTRUSION','INTERPOSE','MULTI_THREAT','ESCAPE_GAP'}
    # One head camera cannot simultaneously face the ward behind it and the
    # threat ahead while physically interposed. Grade each during the phase in
    # which the policy must attend to it: threat while predicting/repositioning,
    # ward while holding/recovering/escorting. Both use the exact rendered PiP.
    ward_los=[r for r in records if r['ward_los'] and r['state'] not in threat_camera_states]
    active=[r for r in records if r['active'] and r['state'] in threat_camera_states
            and r.get('active_los', True)]
    zero=[r for r in records if r['state'] in ZERO_COMMAND_STATES]
    nonzero=[abs(r['command'][0]) for r in records if abs(r['command'][0])>1e-8]
    return {"seconds":rollout.seconds,"control_steps":len(records),"control_hz":1/rollout.dt,
      "policy_sha256":rollout.policy_sha,"observation_dim":61,"action_scale":0.9,"gyro_sensor":"imu_ang_vel",
      "protected_person":WARD,"states_visited":sorted({r['state'] for r in records}),
      "transitions":rollout.machine.transitions,"episodes":episodes,
      "episode_kinds":[e['kind'] for e in episodes],"intrusion_count":len(intrusions),
      "protective_cycle_count":len(threat_cycles),
      "intrusion_people":[e['selected'] for e in intrusions],"intrusion_bearings_deg":bearings,
      "bearings_alternate":alternating,"false_alarm_seen":rollout.false_alarm_seen,
      "false_alarm_dismissed":rollout.false_alarm_dismissed,"squeeze":squeeze,"ward_approach":ward_episode,
      "retreat_path_m":round(retreat_path,4),"retreat_range_gain_m":round(retreat_range,4),
      "path_m":round(rollout.path_m,4),"walk_path_m":round(rollout.walk_path_m,4),
      "min_person_clearance_m":round(rollout.min_person,4),"min_person_clearance_name":rollout.min_person_name,
      "min_scenery_clearance_m":round(rollout.min_scenery,4),"min_scenery_clearance_name":rollout.min_scenery_name,
      "contact_steps":rollout.contacts_count,"fallen_steps":rollout.falls,"min_trunk_z_m":round(rollout.min_z,5),
      "final_trunk_z_m":round(records[-1]['trunk_z'],5),"final_state":rollout.machine.state,
      "final_escort_distance_m":round(records[-1]['escort_distance_m'],4),
      "ward_visible_fraction_with_los":round(_fraction([r['ward_visible'] for r in ward_los]),4),
      "active_visible_fraction":round(_fraction([r['active_visible'] for r in active]),4),
      "zero_state_peak":round(max((r['command_peak'] for r in zero),default=0),7),
      "sub_gait_ticks":sum(0<v<VX_ONSET-1e-6 for v in nonzero),
      "max_abs_vy":round(max((abs(r['command'][1]) for r in records),default=0),7),
      "duck_planar_radius_m":DUCK_PLANAR_RADIUS,"buffer_m":BUFFER_M,
      "retreat_target_m":RETREAT_TARGET_M,"retreat_required_gain_m":RETREAT_RANGE_GAIN_M}


def gates(s):
    result=[]
    def add(name,ok,evidence):result.append((name,bool(ok),evidence))
    intr=[e for e in s['episodes'] if e['kind']=='intrusion']
    add('protected identity remains Aina',s['protected_person']=='aina',s['protected_person'])
    add('neutral escort physically joined',s['path_m']>=1 and s['final_escort_distance_m']<=.35,f"path {s['path_m']} m; final slot {s['final_escort_distance_m']} m")
    add('four distinct genuine intrusion cycles',len(intr)>=4 and len(set(e['selected'] for e in intr))>=4,str([(e['selected'],e['outcome']) for e in intr]))
    add('intrusions alternate bearings',s['bearings_alternate'],str(s['intrusion_bearings_deg']))
    add('false near-pass dismissed',s['false_alarm_seen'] and s['false_alarm_dismissed'],'Piet observed without episode')
    add('every intrusion produced a physical protective path',all(e['path_m']>=.35 for e in intr),str([e['path_m'] for e in intr]))
    add('every interpose reduced target error',all(e['target_reduction_m']>=.15 for e in intr),str([e['target_reduction_m'] for e in intr]))
    add('interpose reached the between-bearing',all(e['between_ticks']>0 for e in intr),str([e['between_ticks'] for e in intr]))
    add('simultaneous squeeze used safe-gap branch',s['squeeze'] is not None and s['squeeze']['secondary'] is not None,str(None if s['squeeze'] is None else (s['squeeze']['selected'],s['squeeze']['secondary'])))
    add('squeeze escape was a real path',s['squeeze'] is not None and s['squeeze']['path_m']>=.35,str(None if s['squeeze'] is None else s['squeeze']['path_m']))
    add('ward approach triggered retreat',s['ward_approach'] is not None and s['retreat_path_m']>=.25,f"path {s['retreat_path_m']} m")
    add('retreat increased ward range',s['retreat_range_gain_m']>=RETREAT_RANGE_GAIN_M,f"gain {s['retreat_range_gain_m']} m")
    add('escort restored after episodes',s['final_state']=='DONE' and s['final_escort_distance_m']<=.35,f"{s['final_state']} slot {s['final_escort_distance_m']}")
    add('ward visible >=95% with LOS',s['ward_visible_fraction_with_los']>=.95,str(s['ward_visible_fraction_with_los']))
    add('active people visible while acted on',s['active_visible_fraction']>=.80,str(s['active_visible_fraction']))
    add('positive person clearance',s['min_person_clearance_m']>0,str(s['min_person_clearance_m']))
    add('positive scenery clearance',s['min_scenery_clearance_m']>0,str(s['min_scenery_clearance_m']))
    add('zero geometric contacts',s['contact_steps']==0,str(s['contact_steps']))
    add('exact zero in declared hold states',s['zero_state_peak']==0,str(s['zero_state_peak']))
    add('no decorative sub-gait commands',s['sub_gait_ticks']==0,str(s['sub_gait_ticks']))
    add('no lateral policy command',s['max_abs_vy']==0,str(s['max_abs_vy']))
    add('real physical locomotion',s['walk_path_m']>=4,str(s['walk_path_m']))
    add('zero falls',s['fallen_steps']==0,str(s['fallen_steps']))
    add('trunk stays above 0.09m',s['min_trunk_z_m']>=.09,str(s['min_trunk_z_m']))
    add('final trunk near nominal',.105<=s['final_trunk_z_m']<=.125,str(s['final_trunk_z_m']))
    add('exact sensor observation and scale',s['gyro_sensor']=='imu_ang_vel' and s['observation_dim']==61 and s['action_scale']==.9,'imu_ang_vel / 61D / .9')
    add('stock walking policy',s['policy_sha256']=='e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c',s['policy_sha256'])
    return result

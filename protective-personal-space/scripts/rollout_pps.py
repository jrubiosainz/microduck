#!/usr/bin/env python3
"""Physical Protective Personal Space rollout.

Scripted people never react to the robot. The policy consumes only measured
poses, constant-velocity predictions and exact PiP visibility; the authored
encounter schedule is not imported here.
"""
from __future__ import annotations
import hashlib, math
from pathlib import Path
import mujoco
import numpy as np
from contact_geometry import ContactProbe, WallProbe, duck_planar_radius
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene
from pps_actors import bodies_at, pose_bodies
from pps_camera import PpsCamera
from pps_cast import ALL_NAMES, WARD
from pps_control import PpsController
from pps_geometry import (escort_point, escape_point, interpose_point,
                          is_between, projected_along, route_around_ward,
                          surface_gap)
from pps_machine import PpsMachine
from pps_plaza import occluder_between
from pps_states import (BUFFER_CLEAR_M, BUFFER_M, ESCORT_JOIN_M, ESCORT_HOLD_S,
                        INTERPOSE_ON_STATION_M, PERSON_APPROACH_CONFIRM_S,
                        PERSON_APPROACH_M, RETREAT_RANGE_GAIN_M,
                        RETREAT_TARGET_M, ZERO_COMMAND_STATES)
from pps_threat import predict_all, priority, squeeze_pair

SCENERY_PREFIXES=("obs_","wall_")


def scenery_names(model):
    return tuple(name for i in range(model.ngeom)
                 if (name:=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i))
                 and name.startswith(SCENERY_PREFIXES))


class PpsRollout:
    def __init__(self, policy_path, seconds=150.0):
        self.model=load_scene(); self.data=mujoco.MjData(self.model)
        self.policy=PolicyRunner(policy_path); self.runner=self.policy.reset(self.model,self.data)
        self.policy_sha=hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
        from pps_plaza import DUCK_START,DUCK_START_YAW_DEG
        self.data.qpos[0:2]=DUCK_START
        yaw=math.radians(DUCK_START_YAW_DEG); self.data.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
        self.trunk=self.model.body("trunk_base").id
        self.dt=1/CTRL_HZ; self.decimation=round(self.dt/self.model.opt.timestep)
        self.seconds=float(seconds); self.total_steps=int(seconds*CTRL_HZ)
        states=bodies_at(0); pose_bodies(self.model,self.data,states,0); mujoco.mj_forward(self.model,self.data)
        self.camera=PpsCamera(self.model,self.data,self.runner.qpos_idx,self.trunk)
        self.machine=PpsMachine(); self.controller=PpsController()
        self.contacts=ContactProbe(self.model,self.trunk,ALL_NAMES,prefix="actor_")
        self.scenery=WallProbe(self.model,self.trunk,scenery_names(self.model))
        self.records=[]; self.previous_states=states; self.previous_xy=self.data.xpos[self.trunk][:2].copy()
        self.path_m=0.; self.walk_path_m=0.; self.min_z=float(self.data.xpos[self.trunk][2]); self.falls=0
        self.min_person=float("inf"); self.min_person_name=""; self.min_scenery=float("inf"); self.min_scenery_name=""; self.contacts_count=0
        self.escort_hold=0.; self.confirm={n:0. for n in ALL_NAMES if n!=WARD}; self.ward_close=0.
        self.false_alarm_seen=False; self.false_alarm_dismissed=False
        self.retreat_start=None; self.retreat_heading=None; self.retreat_range=None
        self.route=[]; self.route_cursor=0
        self.episode_stats={}; self.last_episode_count=0
        self.camera_state=self.camera.update(self.data,self.runner.yaw(self.data),WARD)

    def _place(self,name,xy,z=.008):
        mocap=int(self.model.body_mocapid[self.model.body(name).id])
        self.data.mocap_pos[mocap]=(0,0,-3) if xy is None else (float(xy[0]),float(xy[1]),z)

    def _markers(self,ward,target,active,predictions):
        self._place("ward_buffer",ward.pos); self._place("escort_target",escort_point(ward.pos,ward.yaw))
        self._place("interpose_target",target if self.machine.state in ("INTERPOSE","HOLD_BUFFER") else None)
        self._place("escape_target",target if self.machine.state=="ESCAPE_GAP" else None)
        self._place("active_threat",None if active is None else self.previous_states[active].pos)
        for i in range(16):
            point=None
            if i<len(predictions):
                p=predictions[i]; state=self.previous_states[p.name]
                point=state.pos+state.velocity*min(p.ttc_s,4.)
            self._place(f"prediction_{i}",point)

    def step(self,index):
        t=index*self.dt; duck0=self.data.xpos[self.trunk][:2].copy(); yaw0=self.runner.yaw(self.data)
        ward=self.previous_states[WARD]; people={n:s for n,s in self.previous_states.items() if n!=WARD and s.present}
        predictions=predict_all(ward,people,exclude=self.machine.handled)
        for p in predictions:
            previously_live=self.confirm[p.name]>0.0
            candidate=p.intrusion or (previously_live and p.range_m<BUFFER_M)
            self.confirm[p.name]=self.confirm[p.name]+self.dt if candidate else 0.
        live=[p for p in predictions if (p.intrusion or p.range_m<BUFFER_M)
              and self.confirm[p.name]>=.60]
        # A distant, high-TTC threat is observed longer before committing. This
        # leaves time to recognize a second converging person as a squeeze,
        # rather than charging the first side and discovering the pinch late.
        mature=[p for p in live if p.ttc_s<=5.5 or self.confirm[p.name]>=2.0]
        chosen=priority(mature); pair=squeeze_pair(live)
        ward_range=float(np.linalg.norm(ward.pos-duck0))
        ward_closing=0.0
        if self.records: ward_closing=(self.records[-1]["ward_range_m"]-ward_range)/self.dt
        self.ward_close=(self.ward_close+self.dt if
            t>=77.0 and ward_range<0.72 and ward.speed>0.02
            and WARD not in self.machine.handled else 0.)
        slot=escort_point(ward.pos,ward.yaw); slot_dist=float(np.linalg.norm(slot-duck0))
        escort_joined=slot_dist<=max(ESCORT_JOIN_M,0.35)
        self.escort_hold=self.escort_hold+self.dt if escort_joined else 0.
        escort_ready=self.escort_hold>=min(ESCORT_HOLD_S,0.50)
        target=None; threat_input=None; squeeze_input=None
        if chosen:
            target=interpose_point(ward.pos,self.previous_states[chosen.name].pos)
            threat_input=(chosen.name,target,chosen.record())
        if pair:
            first,second,separation=pair
            locations=[self.previous_states[first.name].pos,self.previous_states[second.name].pos]
            point,score,_=escape_point(ward.pos,locations,{n:s.pos for n,s in people.items()},start=duck0)
            squeeze_input=(first.name,second.name,point,{"bearing_deg":round(first.bearing_deg,2),"secondary_bearing_deg":round(second.bearing_deg,2),"separation_deg":round(separation,2),"gap_score":round(score,4)})
        if self.machine.target is not None: target=np.asarray(self.machine.target)
        # Keep the final between-station attached to the moving ward/threat;
        # only the last route waypoint moves, so the cursor remains monotonic.
        if self.machine.state=="INTERPOSE" and self.machine.selected in self.previous_states:
            moving_target=interpose_point(ward.pos,self.previous_states[self.machine.selected].pos)
            self.machine.target=[float(v) for v in moving_target]
            if self.route: self.route[-1]=moving_target
            target=moving_target
        # Intermediate route corners use a broad capture radius, but the final
        # interpose/escape station must meet its actual on-station tolerance.
        reached=target is not None and float(np.linalg.norm(target-duck0))<=INTERPOSE_ON_STATION_M
        threat_clear=True
        if self.machine.selected and self.machine.selected!=WARD:
            s=self.previous_states[self.machine.selected]
            threat_clear=float(np.linalg.norm(s.pos-ward.pos))>=BUFFER_CLEAR_M
        retreat_complete=False
        if self.retreat_start is not None:
            backed=-projected_along(self.retreat_start,duck0,self.retreat_heading)
            retreat_complete=(backed>=RETREAT_TARGET_M-.05 and ward_range>=self.retreat_range+RETREAT_RANGE_GAIN_M)
        before=self.machine.state
        state,changed=self.machine.update(t,escort_joined=escort_ready,threat=threat_input,
            squeeze=squeeze_input,ward_approach=self.ward_close>=PERSON_APPROACH_CONFIRM_S,
            target_reached=reached,threat_clear=threat_clear,retreat_complete=retreat_complete,
            finish=t>=self.seconds-2.)
        if changed and state in ("INTERPOSE","ESCAPE_GAP"):
            final=np.asarray(self.machine.target)
            self.route=(route_around_ward(duck0,ward.pos,final,heading=yaw0)
                        if state=="INTERPOSE" else [final])
            self.route_cursor=0; target=self.route[0]; self.controller.reset()
        if changed and state=="RETREAT":
            self.retreat_start=duck0.copy(); self.retreat_heading=yaw0; self.retreat_range=ward_range; self.controller.reset()
        if changed and state=="MONITOR" and before in ("RECOVER","RETURN_ESCORT"):
            self.retreat_start=self.retreat_heading=self.retreat_range=None; self.escort_hold=0.
        if state in ("ESCORT","MONITOR","RETURN_ESCORT","RECOVER"): target=slot
        elif state in ("INTERPOSE","ESCAPE_GAP"):
            if not self.route: self.route=[np.asarray(self.machine.target)]; self.route_cursor=0
            if (self.route_cursor < len(self.route)-1 and
                    float(np.linalg.norm(self.route[self.route_cursor]-duck0))<=.38):
                self.route_cursor+=1
            target=self.route[self.route_cursor]
        else: target=None
        command=self.controller.update(state,duck0,yaw0,target_xy=target,
            settle=(target is not None and float(np.linalg.norm(target-duck0))<.30),
            retreat_heading=self.retreat_heading)
        self.runner.step(self.data,command)
        for _ in range(self.decimation): mujoco.mj_step(self.model,self.data)
        display_t=min(t+self.dt,self.seconds); states=bodies_at(display_t); pose_bodies(self.model,self.data,states,display_t)
        duck=self.data.xpos[self.trunk][:2].copy(); z=float(self.data.xpos[self.trunk][2]); travelled=float(np.linalg.norm(duck-self.previous_xy))
        self.path_m+=travelled
        if float(np.max(np.abs(command)))>0: self.walk_path_m+=travelled
        self.min_z=min(self.min_z,z); self.falls+=int(z<.09)
        active=self.machine.selected if self.machine.selected in states else None
        self._markers(states[WARD],target,active,predictions); mujoco.mj_forward(self.model,self.data)
        threat_camera_states=("PREDICT_INTRUSION","INTERPOSE","MULTI_THREAT","ESCAPE_GAP")
        subject=active if active and state in threat_camera_states else WARD
        cam=self.camera.update(self.data,self.runner.yaw(self.data),subject,
                               secondary=WARD if subject!=WARD else None)
        self.camera_state=cam
        clearances={n:self.contacts.distance(self.data,n) for n in ALL_NAMES}; nearest=min(clearances,key=clearances.get)
        scenery,scenery_name=self.scenery.distance(self.data)
        if clearances[nearest]<self.min_person:self.min_person,self.min_person_name=clearances[nearest],nearest
        if scenery<self.min_scenery:self.min_scenery,self.min_scenery_name=scenery,scenery_name
        self.contacts_count+=int(clearances[nearest]<=0 or scenery<=0)
        ward_now=states[WARD]; wrange=float(np.linalg.norm(ward_now.pos-duck)); active_state=states.get(active) if active else None
        actual_threat_range=None if active_state is None else float(np.linalg.norm(active_state.pos-ward_now.pos))
        between=False if active_state is None else is_between(ward_now.pos,duck,active_state.pos)
        ward_los=self.camera.has_line_of_sight(WARD)
        active_los=(False if active_state is None else
                    self.camera.has_line_of_sight(active))
        protective_target=(None if self.machine.target is None else
                           float(np.linalg.norm(np.asarray(self.machine.target)-duck)))
        record={"t":round(display_t,3),"state":state,"command":[float(v) for v in command],"command_peak":float(np.max(np.abs(command))),
            "duck_xy":duck.tolist(),"duck_yaw_deg":math.degrees(self.runner.yaw(self.data)),"trunk_z":z,"path_m":self.path_m,
            "ward_xy":ward_now.pos.tolist(),"ward_range_m":wrange,"ward_visible":cam["people"][WARD]["visible"],"ward_los":ward_los,
            "active":active,"active_visible":False if active is None else cam["people"][active]["visible"],"active_los":active_los,"threat_range_m":actual_threat_range,
            "target":None if target is None else np.asarray(target).tolist(),"target_distance_m":None if target is None else float(np.linalg.norm(np.asarray(target)-duck)),"protective_target_distance_m":protective_target,
            "between":between,"predictions":[p.record() for p in predictions[:5]],"escort_distance_m":float(np.linalg.norm(slot-duck)),
            "min_person_clearance_m":clearances[nearest],"nearest_person":nearest,"scenery_clearance_m":scenery,"nearest_scenery":scenery_name}
        if self.machine.state=="MONITOR" and any(p.name=="piet" and p.range_m<3.5 for p in predictions): self.false_alarm_seen=True
        if self.false_alarm_seen and "piet" not in [e.get("selected") for e in self.machine.episodes]: self.false_alarm_dismissed=True
        self.records.append(record); self.previous_states=states; self.previous_xy=duck.copy(); return record

    def run(self,progress=None,on_frame=None):
        # ``on_frame`` is a RENDER-ONLY observer. It is invoked AFTER step(i)
        # has completed, receives the same record that was just appended, and
        # returns nothing that is read back. The deterministic sequence -
        # policy inference, mj_step, actor posing, camera update, record - is
        # identical whether or not a callback is attached, which is what lets
        # the rendered run be graded by the same gates as the headless one.
        for i in range(self.total_steps):
            r=self.step(i)
            if on_frame is not None: on_frame(i,r)
            if progress and i%250==0: progress(i,r)
        self.machine.finish(self.seconds); return self.records

"""Spot fleet keeper that chases price per game: picks (instance type, AZ) pools
by dollars per 100k self-play games, launches the instances itself (no Auto
Scaling group), registers them in the ECS cluster, places one actor task per
box for the arm that is furthest below its share, falls back to the next pool
when the one it aims for fails capacity several times in a row, and migrates
off a box once a clearly cheaper pool has been available for a while.

    .venv/bin/python runs/fleet.py --dry-run      # read-only: prices, plan, what it would do
    .venv/bin/python runs/fleet.py --once         # one reconcile cycle
    nohup .venv/bin/python runs/fleet.py > runs/fleet.out 2>&1 &

Config: runs/fleet.json (arms and their shares, type throughput table, caps).
State:  runs/fleet_state.json (pool failures, price cache, migration timers; also what a dashboard reads).
The old keeper (runs/ax_keeper.sh) and the stack's Auto Scaling group must be
stopped / set to max 0 before this runs, or both will provision actors.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

import boto3
from botocore.exceptions import ClientError

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG, STATE = "runs/fleet.json", "runs/fleet_state.json"
FLEET_TAG = "catan:fleet"
CAPACITY_ERRORS = {"InsufficientInstanceCapacity", "SpotMaxPriceTooLow", "MaxSpotInstanceCountExceeded", "Unsupported",
                   "InstanceLimitExceeded", "VcpuLimitExceeded", "InsufficientCapacityOnHost", "SpotFleetRequestConfigurationInvalid"}


def now():
    return time.time()


def stamp():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"{stamp()} {msg}", flush=True)


class Fleet:
    def __init__(self, cfg, dry_run):
        self.cfg, self.dry = cfg, dry_run
        region = cfg.get("region", "us-east-1")
        self.ec2, self.ecs = boto3.client("ec2", region_name=region), boto3.client("ecs", region_name=region)
        self.state = json.load(open(STATE)) if os.path.exists(STATE) else {}
        self.state.setdefault("pools", {})        # "type@az" -> {"fails": n, "cool_until": ts}
        self.state.setdefault("launched", {})     # instance id -> {"pool": .., "at": ts, "usd_per_100k": ..}
        self.state.setdefault("overpriced", {})   # instance id -> since ts
        self.state.setdefault("idle", {})         # instance id -> since ts
        self.state.setdefault("last_migration", 0)
        self.state.setdefault("handover", {})     # replacement instance id -> old instance id it retires once it runs a task
        self.state.setdefault("prices", {"at": 0, "pools": []})
        self.vcpus = {t["InstanceType"]: t["VCpuInfo"]["DefaultVCpus"]
                      for t in self.ec2.describe_instance_types(InstanceTypes=list(cfg["types"]))["InstanceTypes"]}
        self.subnets = {s["AvailabilityZone"]: s["SubnetId"] for s in self.ec2.describe_subnets(
            Filters=[{"Name": "tag:aws-cdk:subnet-type", "Values": ["Public"]},
                     {"Name": "tag:aws:cloudformation:stack-name", "Values": [cfg["stack"]]}])["Subnets"]}

    def save(self):
        tmp = STATE + ".tmp"
        json.dump(self.state, open(tmp, "w"), indent=1)
        os.replace(tmp, STATE)

    # ---- what the arms want
    def arms_wanted(self):
        """Arm -> share in units (one unit = a Zen 4 16xlarge actor); arms whose learner isn't running want 0,
        except those listed in config "force_active" (to have a box ready before a batch of short learners starts)."""
        learners = subprocess.run("ps -eo args | grep -o 'az.py --run [^ ]*' | awk '{print $3}'", shell=True,
                                  capture_output=True, text=True).stdout.split()
        forced = set(self.cfg.get("force_active", []))
        # A learner that vanished keeps its share for a grace period: restarts (resume, crash, flag change) take
        # minutes, and tearing the boxes down meanwhile costs a relaunch and the capacity hunt (it did once).
        seen = self.state.setdefault("learner_seen", {})
        grace = self.cfg.get("arm_grace_minutes", 20) * 60
        out = {}
        for arm, units in self.cfg["arms"].items():
            if any(l == arm or l.startswith(arm + "-") for l in learners):
                seen[arm] = now()
            active = arm in forced or now() - seen.get(arm, 0) < grace
            out[arm] = units if active else 0
        return out

    # ---- prices and pools
    def pools(self):
        """Eligible (type, az) pools with the latest spot price, sorted by dollars per 100k games."""
        cache = self.state["prices"]
        if now() - cache["at"] > self.cfg["price_poll_minutes"] * 60:
            latest = {}
            start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=3)
            pages = self.ec2.get_paginator("describe_spot_price_history").paginate(
                InstanceTypes=list(self.cfg["types"]), ProductDescriptions=["Linux/UNIX"], StartTime=start)
            for page in pages:
                for h in page["SpotPriceHistory"]:
                    k = (h["InstanceType"], h["AvailabilityZone"])
                    if k not in latest or h["Timestamp"] > latest[k][0]:
                        latest[k] = (h["Timestamp"], float(h["SpotPrice"]))
            cache["pools"] = [{"type": t, "az": az, "price": p, "gph": self.cfg["types"][t], "vcpu": self.vcpus[t],
                               "usd_per_100k": p / self.cfg["types"][t] * 1e5}
                              for (t, az), (_, p) in latest.items() if az in self.subnets]
            cache["at"] = now()
        out = []
        for p in cache["pools"]:
            st = self.state["pools"].get(f"{p['type']}@{p['az']}", {})
            p = dict(p, cooling=st.get("cool_until", 0) > now(), fails=st.get("fails", 0))
            p["eligible"] = p["usd_per_100k"] <= self.cfg["max_usd_per_100k_games"] and not p["cooling"]
            out.append(p)
        return sorted(out, key=lambda p: p["usd_per_100k"])

    def pool_failed(self, pool, why):
        st = self.state["pools"].setdefault(pool, {"fails": 0, "cool_until": 0})
        st["fails"] += 1
        st["last_fail"] = now()
        if st["fails"] >= self.cfg["fail_limit"]:
            st["cool_until"] = now() + self.cfg["cooldown_minutes"] * 60
            st["fails"] = 0
            log(f"pool {pool}: {why}; {self.cfg['fail_limit']} failures in a row, cooling for {self.cfg['cooldown_minutes']} min")
        else:
            log(f"pool {pool}: {why} (failure {st['fails']} of {self.cfg['fail_limit']})")

    def pool_ok(self, pool):
        st = self.state["pools"].setdefault(pool, {"fails": 0, "cool_until": 0})
        st["fails"], st["last_success"] = 0, now()

    def pool_proven(self, pool, need_success):
        """A migration target must have been quiet (no capacity failure) for a while and, when we'd have to free
        the old box's vCPUs before launching, must have actually given us a box recently."""
        st = self.state["pools"].get(pool, {})
        quiet = now() - st.get("last_fail", 0) > self.cfg.get("migrate_pool_quiet_minutes", 60) * 60
        fresh = now() - st.get("last_success", 0) < self.cfg.get("migrate_pool_success_minutes", 120) * 60
        return quiet and (fresh or not need_success)

    # ---- inventory
    def inventory(self):
        """Our instances, every spot vCPU in the account (quota), ECS container instances and running tasks."""
        ours, foreign_vcpus = {}, 0
        pages = self.ec2.get_paginator("describe_instances").paginate(
            Filters=[{"Name": "instance-state-name", "Values": ["pending", "running"]}])
        for page in pages:
            for r in page["Reservations"]:
                for i in r["Instances"]:
                    tags = {t["Key"]: t["Value"] for t in i.get("Tags", [])}
                    if tags.get(FLEET_TAG):
                        ours[i["InstanceId"]] = {"type": i["InstanceType"], "az": i["Placement"]["AvailabilityZone"],
                                                 "state": i["State"]["Name"], "launched": i["LaunchTime"].timestamp(),
                                                 "pool": f"{i['InstanceType']}@{i['Placement']['AvailabilityZone']}"}
                    elif i.get("InstanceLifecycle") == "spot":
                        foreign_vcpus += i["CpuOptions"]["CoreCount"] * i["CpuOptions"]["ThreadsPerCore"]
        cis = {}
        arns = self.ecs.list_container_instances(cluster=self.cfg["cluster"])["containerInstanceArns"]
        if arns:
            for ci in self.ecs.describe_container_instances(cluster=self.cfg["cluster"], containerInstances=arns)["containerInstances"]:
                cis[ci["ec2InstanceId"]] = {"arn": ci["containerInstanceArn"], "status": ci["status"], "agent": ci["agentConnected"],
                                            "running": ci["runningTasksCount"], "pending": ci["pendingTasksCount"]}
        tasks = []
        for status in ("RUNNING", "PENDING"):
            tarns = self.ecs.list_tasks(cluster=self.cfg["cluster"], desiredStatus=status)["taskArns"]
            if tarns:
                for t in self.ecs.describe_tasks(cluster=self.cfg["cluster"], tasks=tarns)["tasks"]:
                    cmd = (t.get("overrides", {}).get("containerOverrides") or [{}])[0].get("command") or []
                    arm = next((c[len("models/control_"):-len(".json")] for c in cmd if c.startswith("models/control_")), None)
                    tasks.append({"arn": t["taskArn"], "ci": t["containerInstanceArn"], "arm": arm, "status": t["lastStatus"]})
        ci_to_ec2 = {v["arn"]: k for k, v in cis.items()}
        for t in tasks:
            t["instance"] = ci_to_ec2.get(t["ci"])
        return ours, foreign_vcpus, cis, tasks

    # ---- actions
    def launch(self, pool):
        user_data = ("#!/bin/bash\n" f"echo ECS_CLUSTER={self.cfg['cluster']} >> /etc/ecs/ecs.config\n"
                     "echo ECS_ENABLE_SPOT_INSTANCE_DRAINING=true >> /etc/ecs/ecs.config\n")
        max_price = self.cfg["max_usd_per_100k_games"] * pool["gph"] / 1e5  # the same $/game cap, as $/hour for this type
        name = f"{pool['type']}@{pool['az']}"
        if self.dry:
            log(f"would launch {name} at ${pool['price']:.3f}/h (${pool['usd_per_100k']:.2f}/100k games, max ${max_price:.2f}/h)")
            return None
        # Graviton families (c8g, m8g, r8g, c7g...) use the stack's ARM launch template (ARM ECS AMI); the image is multi-arch.
        family = pool["type"].split(".")[0]
        arch = "arm" if family.rstrip("d").endswith("g") else "x86"
        templates = self.cfg.get("launch_templates", {"x86": self.cfg["launch_template"]["id"]})
        if arch not in templates:
            self.pool_failed(name, f"no {arch} launch template configured")
            return False
        try:
            r = self.ec2.run_instances(
                LaunchTemplate={"LaunchTemplateId": templates[arch], "Version": self.cfg["launch_template"]["version"]},
                InstanceType=pool["type"], MinCount=1, MaxCount=1, SubnetId=self.subnets[pool["az"]], UserData=user_data,
                InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {"MaxPrice": f"{max_price:.4f}", "SpotInstanceType": "one-time",
                                                                              "InstanceInterruptionBehavior": "terminate"}},
                TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": FLEET_TAG, "Value": "1"}, {"Key": "Name", "Value": f"catan-fleet-{pool['type']}"},
                                                                         {"Key": "catan:pool", "Value": name}]}])
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("VcpuLimitExceeded", "MaxSpotInstanceCountExceeded", "InstanceLimitExceeded"):
                log(f"launch of {name} refused by our own quota ({code}); not a pool failure")
                return False
            if code in CAPACITY_ERRORS:
                self.pool_failed(name, code)
                return False
            raise
        iid = r["Instances"][0]["InstanceId"]
        self.pool_ok(name)
        self.state["launched"][iid] = {"pool": name, "at": now(), "usd_per_100k": pool["usd_per_100k"]}
        log(f"launched {iid} {name} at ${pool['price']:.3f}/h (${pool['usd_per_100k']:.2f}/100k games)")
        return iid

    def start_task(self, ci_arn, arm, iid):
        if self.dry:
            log(f"would start {arm}'s actor on {iid}")
            return
        r = self.ecs.start_task(cluster=self.cfg["cluster"], taskDefinition=self.cfg["task_definition"], containerInstances=[ci_arn],
                                overrides={"containerOverrides": [{"name": "actor", "command": ["python", "actor.py", "--control", f"models/control_{arm}.json"]}]})
        if r["failures"]:
            log(f"start_task on {iid} for {arm} failed: {r['failures']}")
        else:
            log(f"started {arm}'s actor on {iid}")

    def stop_task(self, task, why):
        if self.dry:
            log(f"would stop {task['arm']}'s task on {task['instance']}: {why}")
            return
        self.ecs.stop_task(cluster=self.cfg["cluster"], task=task["arn"], reason=why)
        log(f"stopped {task['arm']}'s task on {task['instance']}: {why}")

    def terminate(self, iid, why):
        if self.dry:
            log(f"would terminate {iid}: {why}")
            return
        self.ec2.terminate_instances(InstanceIds=[iid])
        log(f"terminated {iid}: {why}")
        self.state["launched"].pop(iid, None)
        self.state["overpriced"].pop(iid, None)
        self.state["idle"].pop(iid, None)

    # ---- one reconcile cycle
    def cycle(self):
        self.cfg = json.load(open(CONFIG))  # shares, caps and force_active can be edited while it runs
        cfg = self.cfg
        wanted = self.arms_wanted()
        unit = cfg["unit_games_per_hour"]
        target_gph = sum(wanted.values()) * unit
        pools = self.pools()
        price_of = {f"{p['type']}@{p['az']}": p for p in pools}
        ours, foreign_vcpus, cis, tasks = self.inventory()

        # Boxes that vanished soon after launch: the pool reclaimed them (counts as a capacity failure).
        for iid, rec in list(self.state["launched"].items()):
            if iid not in ours:
                if now() - rec["at"] < 15 * 60:
                    self.pool_failed(rec["pool"], "gone within 15 min of launch")
                self.state["launched"].pop(iid)

        # Current supply per arm, in games/hour: tasks on our boxes and on any other box in the cluster (the old group's, while it lasts).
        # The `types` table is in reference games/h (the settings it was measured with); an arm whose games cost more
        # compute (wider inputs, more forks per game) divides it by its `arm_game_cost`, so shares and deficits are in
        # that arm's own games/h. Pool prices and the $/100k cap stay in reference units (the ranking is the same).
        cost = lambda arm: cfg.get("arm_game_cost", {}).get(arm, 1.0) if arm else 1.0
        supply = {arm: 0.0 for arm in wanted}
        task_on = {}
        for t in tasks:
            if t["instance"]:
                task_on[t["instance"]] = t
            typ = ours.get(t["instance"], {}).get("type") or self.cluster_instance_type(t["instance"])
            if t["arm"] in supply and typ in cfg["types"]:
                supply[t["arm"]] += cfg["types"][typ] / cost(t["arm"])
        # A box's rate in the games of the arm it serves (an idle box: the arm furthest below its share, where it would go).
        neediest = max(wanted, key=lambda a: wanted[a] * unit - supply[a]) if wanted else None
        box_rate = lambda iid, inst: cfg["types"][inst["type"]] / cost(task_on[iid]["arm"] if iid in task_on and task_on[iid]["arm"] in wanted else neediest)
        our_gph = sum(box_rate(i, inst) for i, inst in ours.items())
        foreign_gph = sum(supply.values()) - sum(box_rate(i, ours[i]) for i, t in task_on.items() if i in ours and t["arm"] in supply)
        used_vcpus = sum(self.vcpus.get(i["type"], 0) for i in ours.values()) + foreign_vcpus
        free_vcpus = cfg["vcpu_quota"] - used_vcpus

        # 1. Arms whose learner is gone: stop their tasks.
        for t in tasks:
            if t["arm"] is not None and wanted.get(t["arm"], 0) == 0 and t["instance"] in ours:
                self.stop_task(t, f"{t['arm']} has no learner")

        # 2. Place actors on our registered boxes that have none: the arm furthest below its share.
        for iid, inst in ours.items():
            ci = cis.get(iid)
            if not ci or ci["status"] != "ACTIVE" or not ci["agent"] or ci["running"] + ci["pending"] > 0 or iid in task_on:
                continue
            deficit = {arm: wanted[arm] * unit - supply[arm] for arm in wanted if wanted[arm] > 0}
            replaces = self.state["handover"].get(iid)
            if replaces and replaces in task_on and task_on[replaces]["arm"] in wanted:
                arm = task_on[replaces]["arm"]  # a migration's replacement takes over the old box's arm
            elif not deficit:
                continue
            else:
                arm = max(deficit, key=deficit.get)
                if deficit[arm] <= 0 and our_gph + foreign_gph - box_rate(iid, inst) >= target_gph:
                    continue  # nobody needs it; the idle rule below handles it
            self.start_task(ci["arn"], arm, iid)
            supply[arm] += cfg["types"][inst["type"]] / cost(arm)

        # 3. Idle boxes nobody needs for a while: terminate.
        for iid, inst in ours.items():
            busy = iid in task_on or (cis.get(iid, {}).get("running", 0) + cis.get(iid, {}).get("pending", 0) > 0)
            registering = now() - inst["launched"] < 10 * 60 and iid not in cis
            if busy or registering:
                self.state["idle"].pop(iid, None)
                continue
            since = self.state["idle"].setdefault(iid, now())
            if now() - since > cfg["idle_terminate_minutes"] * 60 and (target_gph == 0 or our_gph + foreign_gph - box_rate(iid, inst) >= target_gph):
                self.terminate(iid, f"idle {cfg['idle_terminate_minutes']} min and not needed")

        # 4. An arm below its share (one box serves one arm, so this is per arm): launch the cheapest eligible
        #    pool that fits the quota for the arm furthest below, one launch per cycle. Boxes still booting
        #    count toward the arm they will get (the one furthest below when they registered is the same rule).
        booting = sum(box_rate(i, inst) for i, inst in ours.items() if i not in task_on)
        deficit = {arm: wanted[arm] * unit - supply[arm] for arm in wanted if wanted[arm] > 0}
        short = [arm for arm, d in sorted(deficit.items(), key=lambda kv: -kv[1]) if d > 0]
        if short and booting < sum(deficit[a] for a in short):
            arm = short[0]
            fits = [p for p in pools if p["eligible"] and p["vcpu"] <= free_vcpus]
            best_any = next((p for p in pools if p["eligible"]), None)
            # The best pool overall needs vCPUs that other spot boxes (the old group's, on their way out) still
            # hold: wait a little for them rather than settle for a dearer pool that then blocks the quota.
            quota_without_foreign = cfg["vcpu_quota"] - sum(self.vcpus.get(i["type"], 0) for i in ours.values())
            if best_any and foreign_vcpus > 0 and free_vcpus < best_any["vcpu"] <= quota_without_foreign and (not fits or fits[0] is not best_any):
                since = self.state.setdefault("quota_wait_since", now())
                if now() - since < cfg.get("wait_for_quota_minutes", 10) * 60:
                    log(f"{arm} is short; waiting for {best_any['vcpu'] - free_vcpus} vCPUs held by other spot boxes to free for "
                        f"{best_any['type']}@{best_any['az']} (${best_any['usd_per_100k']:.2f}/100k"
                        + (f" vs ${fits[0]['usd_per_100k']:.2f}/100k for what fits now)" if fits else ")"))
                    fits = []
            else:
                self.state.pop("quota_wait_since", None)
            if fits:
                # Aim at the cheapest; when it has no capacity right now, fall through to the next pools in the same
                # cycle (a few at most) so the arm isn't left idle while the aim's failures accumulate toward its cooldown.
                log(f"{arm} has {supply[arm]:,.0f} of {wanted[arm] * unit:,.0f} games/h; free vCPUs {free_vcpus}; aiming at {fits[0]['type']}@{fits[0]['az']}")
                for pool in fits[:cfg.get("attempts_per_cycle", 3)]:
                    if self.launch(pool) is not False:
                        break
            elif "quota_wait_since" not in self.state:
                log(f"{arm} has {supply[arm]:,.0f} of {wanted[arm] * unit:,.0f} games/h but nothing eligible fits in {free_vcpus} free vCPUs "
                    f"(quota {cfg['vcpu_quota']}, other spot vCPUs {foreign_vcpus})")

        # 5. Handovers: a migration's replacement box is running a task -> retire the box it replaces.
        for new_iid, old_iid in list(self.state["handover"].items()):
            if new_iid not in ours:  # the replacement never came up (reclaimed, failed to boot): keep the old box
                self.state["handover"].pop(new_iid)
            elif new_iid in task_on:
                if old_iid in task_on:
                    self.stop_task(task_on[old_iid], "replaced by a cheaper box")
                if old_iid in ours:
                    self.terminate(old_iid, "replaced by a cheaper box")
                self.state["handover"].pop(new_iid)

        # 6. Migration: a box clearly dearer per game than a proven cheaper pool for a while, one at a time.
        #    Preferred: launch the replacement first (it fits beside the old box) and retire the old one when the
        #    new one runs. Only when the replacement needs the old box's vCPUs do we terminate first, and then only
        #    toward a pool that has given us a box recently and has not failed capacity for an hour.
        if pools and now() - self.state["last_migration"] > cfg["migrate_interval_minutes"] * 60:
            for iid, inst in ours.items():
                cur = price_of.get(inst["pool"])
                if not cur or iid not in task_on or iid in self.state["handover"].values():
                    self.state["overpriced"].pop(iid, None)
                    continue
                gph_box = cfg["types"][inst["type"]]

                def alt_for(room, need_success):
                    return next((p for p in pools if p["eligible"] and p["vcpu"] <= room and p["gph"] >= 0.9 * gph_box
                                 and self.pool_proven(f"{p['type']}@{p['az']}", need_success)), None)
                alt_now = alt_for(free_vcpus, need_success=False)
                # Terminating first is only hedged by a proven pool when the box is itself acceptable; a box that costs
                # more per game than we would launch at (over the cap) is replaced toward any quiet eligible pool.
                over_cap = cur["usd_per_100k"] > cfg["max_usd_per_100k_games"]
                alt_after = alt_for(free_vcpus + self.vcpus[inst["type"]], need_success=not over_cap)
                alt = min((a for a in (alt_now, alt_after) if a), key=lambda p: p["usd_per_100k"], default=None)
                if alt is None or cur["usd_per_100k"] <= alt["usd_per_100k"] * (1 + cfg["migrate_saving"]):
                    self.state["overpriced"].pop(iid, None)
                    continue
                since = self.state["overpriced"].setdefault(iid, now())
                if now() - since < cfg["migrate_after_minutes"] * 60:
                    continue
                log(f"migrating off {iid} {inst['pool']} (${cur['usd_per_100k']:.2f}/100k) toward {alt['type']}@{alt['az']} (${alt['usd_per_100k']:.2f}/100k)")
                if alt is alt_now:
                    new = self.launch(alt)
                    if new:
                        self.state["handover"][new] = iid
                        self.state["last_migration"] = now()
                else:
                    self.stop_task(task_on[iid], "migrating to a cheaper pool")
                    self.terminate(iid, "migrating to a cheaper pool")
                    self.state["last_migration"] = now()
                self.state["overpriced"].pop(iid, None)
                break

        self.state["summary"] = {"at": now(), "wanted": wanted, "target_gph": target_gph, "supply": supply, "free_vcpus": free_vcpus,
                                 "boxes": [{"id": i, **inst, "arm": task_on.get(i, {}).get("arm"),
                                            "usd_per_100k": price_of.get(inst["pool"], {}).get("usd_per_100k")} for i, inst in ours.items()],
                                 "pools": pools[:12]}
        self.save()
        return wanted, target_gph, supply, free_vcpus, ours, pools

    def shutdown(self, why):
        """Stop our tasks and terminate our boxes (the safety deadline)."""
        ours, _, _, tasks = self.inventory()
        for t in tasks:
            if t["instance"] in ours:
                self.stop_task(t, why)
        for iid in ours:
            self.terminate(iid, why)
        self.save()

    _ci_types = {}

    def cluster_instance_type(self, iid):
        """Instance type of a cluster box that isn't ours (cached)."""
        if not iid:
            return None
        if iid not in self._ci_types:
            r = self.ec2.describe_instances(InstanceIds=[iid])["Reservations"]
            self._ci_types[iid] = r[0]["Instances"][0]["InstanceType"] if r else None
        return self._ci_types[iid]


def report(fleet, wanted, target_gph, supply, free_vcpus, ours, pools):
    log(f"arms {wanted} -> target {target_gph:,.0f} games/h; supply {{{', '.join(f'{a}: {g:,.0f}' for a, g in supply.items())}}}; free vCPUs {free_vcpus}")
    for iid, inst in ours.items():
        p = next((p for p in pools if f"{p['type']}@{p['az']}" == inst["pool"]), None)
        log(f"  box {iid} {inst['pool']} {inst['state']}" + (f" ${p['usd_per_100k']:.2f}/100k" if p else ""))
    log("cheapest pools ($ per 100k games; * = eligible):")
    for p in pools[:10]:
        log(f"  {'*' if p['eligible'] else ' '} {p['type']:16s} {p['az']}  ${p['price']:.3f}/h  {p['gph']:>8,} games/h  ${p['usd_per_100k']:.2f}/100k"
            + ("  cooling" if p["cooling"] else "") + (f"  fails {p['fails']}" if p["fails"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="read-only: print prices, plan and the actions it would take")
    ap.add_argument("--once", action="store_true", help="one cycle, then exit")
    ap.add_argument("--interval", type=int, default=60, help="seconds between cycles")
    ap.add_argument("--hours", type=float, default=None,
                    help="safety deadline: stop our tasks and terminate our boxes after this long (default: config deadline_hours)")
    args = ap.parse_args()
    cfg = json.load(open(CONFIG))
    fleet = Fleet(cfg, args.dry_run)
    deadline = now() + (args.hours if args.hours is not None else cfg.get("deadline_hours", 14)) * 3600
    while True:
        if now() > deadline:
            log("safety deadline reached: shutting the fleet down")
            fleet.shutdown("fleet keeper safety deadline")
            return
        try:
            out = fleet.cycle()
            if args.dry_run or args.once:
                report(fleet, *out)
        except Exception as e:  # keep the loop alive through transient API errors
            log(f"cycle error: {type(e).__name__}: {e}")
            if args.dry_run or args.once:
                raise
        if args.dry_run or args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

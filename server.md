# Study servers

Three Oracle Cloud A100 boxes run the consequence-interface-versus-attributes
comparison for the two construction solvers. Everything below is under
`~/CaR-constraint` on the remote, which is a copy of this directory.

| | host | arms | GPU |
|---|---|---|---|
| **M1** | `ubuntu@132.145.182.17` | POMO **attr**, POMO **interface** | A100-SXM4-40GB |
| **M2** | `ubuntu@132.145.197.191` | CaR **attr** | A100-SXM4-40GB |
| **M3** | `ubuntu@129.158.251.32` | CaR **interface** | A100-SXM4-40GB |

The split is now by *solver*: the two fast POMO arms share M1, and each slow CaR
arm has a machine to itself. All three boxes are the same instance type —
A100-SXM4-40GB on an AMD EPYC 7J13 — so an arm-versus-arm difference is never a
hardware difference.

This layout is what makes the CaR comparison clean. Both CaR arms now run alone
on identical hardware, so the epoch rate difference between them is intrinsic to
the representation rather than an artefact of who they were sharing a GPU with.
The POMO arms share a machine with each other, so they too are matched — both
contended, equally.

Layout history, both on 2026-09-07: CaR interface moved M2 -> M3 at **epoch
150** (it was sharing M2 with POMO interface and held to ~29 min/epoch, ~17 days
from epoch 1000). Then CaR attr moved M1 -> M2 at **epoch 205** and POMO
interface moved M2 -> M1 at **epoch 485**, giving the layout above.

Keeping the whole study on one instance type is deliberate. A GH200 was tried
first and abandoned: this workload sits at ~17% GPU utilisation and ~1 busy
core, so it is latency-bound on small batches rather than GPU-bound, and a
bigger GPU buys nothing while the move would have put the two CaR arms on
different GPU models. Match the instance type when adding capacity.

"POMO" is `--improve_steps 0 --pomo_start True --diversity_loss False
--soft_constrained True`; "CaR" (construct-and-refine) is `--improve_steps 5
--validation_improve_steps 20`. Both train on **VRPBLTW**, n=50, seed 1234.

Access is by key, no password: `ssh ubuntu@132.145.182.17`.

## Running now

Four jobs — two on M1, one each on M2 and M3 — all extending to **epoch 1000**
from earlier checkpoints with optimizer state restored (`--load_optimizer
True`).

| job | machine | launcher | nohup log | writes to |
|---|---|---|---|---|
| POMO attr | M1 | `resume1000.sh attr <ckpt>` | `resume1000.log` | `results/pomo1000/train_pomo_soft_attr/` |
| POMO interface | M1 | `resume1000.sh interface <ckpt>` | `resume1000_interface.log` | `results/pomo1000/train_pomo_soft_interface/` |
| CaR attr | M2 | `car1000_attr.sh` | `car1000.log` | `results/car1000/train_attr/` |
| CaR interface | M3 | `car1000_m4.sh` | `car1000_m4.log` | `results/car1000/train_interface/` |

Every moved job kept its original flags: each new launcher was diffed against
the launcher it replaces and only the resume checkpoint differs. Logs of moved
jobs are archived on the source machine as `archive_*_moved-to-*.log` so
`status.sh` does not mistake a moved job for a dead one, and the checkpoints
stay where they were — hence the "checkpoints only" rows it prints.

`--milestones` defaults to `[4501]`, which is past 1000, so the learning rate is
**constant** over the whole range and extending the horizon does not change the
schedule. Checkpoints land every 5 epochs (`--model_save_interval 5`), so the
runs can be stopped at any point and read at a matched epoch.

The two solvers progress at very different rates — CaR is roughly 3x slower per
epoch than POMO, and sharing a GPU slows both further. **Compare at matched
epoch, never at matched wall-clock.** Contention costs time, not validity.

Measured CaR rates, for planning:

| | min/epoch |
|---|---|
| CaR attr, M1, sharing the GPU with POMO attr | ~19 |
| CaR interface, M2, sharing the GPU with POMO interface | ~29 |
| CaR interface, M3, alone | ~19 |

The interface arm is intrinsically slower per epoch than attr — it publishes the
constraint rows — so on a shared GPU it fell behind. Giving it a machine of its
own brings it back to the attr arm's rate, which is what keeps the two CaR arms
progressing in lockstep. At ~19 min/epoch, epoch 1000 is still ~11 days out from
epoch 150.

## Reading progress

`./status.sh` in `~/CaR-constraint` on any of the three machines prints the
whole picture for that box — current epoch, checkpoint age, ETA, the last
validation gaps, the recent gap curve and the soft-construction infeasibility
curve — in an identical layout everywhere, so two outputs can be read side by
side to find a matched epoch.

```sh
ssh ubuntu@132.145.182.17 'cd ~/CaR-constraint && ./status.sh'        # dashboard
ssh ubuntu@132.145.182.17 'cd ~/CaR-constraint && ./status.sh 485'    # one exact epoch
ssh ubuntu@132.145.182.17 'cd ~/CaR-constraint && ./status.sh -n 20'  # longer curve
```

It discovers jobs from the **running processes** rather than a hardcoded layout:
solver and arm come from each process's own command line, so it stays correct
however the jobs are spread across machines — including M1 running two POMO arms
at once. Run directories with checkpoints but no live process are listed as
"checkpoints only", which is what a moved job looks like from its old machine.

Passing an epoch looks it up in every job on that box and warns when the
matching checkpoint is absent, which is what `matched_eval.sh` needs. Validation
runs every 5 epochs, so only multiples of 5 resolve; anything else reports the
nearest below.

### Do not read validation numbers from the nohup log

**The logs are useless for live monitoring.** Python block-buffers stdout at
8 KB when redirected, and this trainer prints so little per epoch that the
buffer takes hours to fill: M3's `car1000_m4.log` had grown to 10 KB after
3.5 hours of training and contained **zero** `Val Score` lines while the run was
already past two validations. This is not a lag of minutes, and it is not a
stall — the numbers simply are not in the file yet.

The TensorBoard event files are flushed every epoch, so read those instead.
`read_val.py` does it, and `status.sh` uses it:

```sh
ssh ubuntu@129.158.251.32 'cd ~/CaR-constraint && python3 read_val.py results/car1000/train_interface'
# 155 11.7841 12.1244 12.5000 11.9614      <- epoch masked constr infeas improve
```

It merges scalars across every timestamped run directory under the `log_dir`, so
the curve spans resumes instead of restarting at each move.

The tags were verified against the log lines they mirror, at epoch 145 of the
CaR interface arm:

| tag | log line | value |
|---|---|---|
| `val/val_gap_rc_masked` | `[w. mask] AUG_Gap` | 11.7671 |
| `val/val_gap` | `[Construction] AUG_Gap` | 12.2829 |
| `val/improve_val_gap` | `[Improvement] AUG_Gap` | 11.9588 |
| `val/val_sol_infsb_rate` | `[Construction]` infeasible rate | 16.309 |

`val/val_gap_rc_masked` is the masked-pass objective — the only one feasible for
every arm in every variant, and the number reported as `Gap (%)` in
`tables/interface_solvers.tex`. The historical logs still contain these lines
once flushed, and completed runs can be read either way.

Rate and ETA in `status.sh` are measured from checkpoint mtimes over the last
few checkpoints, not from the log's own `Remain[...]` estimate — jobs get moved
and change GPU neighbours, so a lifetime average reports a speed the machine is
no longer running at.

Liveness:

```sh
ssh ubuntu@132.145.182.17 'pgrep -af "trai[n].py"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
```

Note the bracket in `"trai[n].py"`. Patterns that match their own command string
will kill or match the ssh session itself — this has already cost one session
(exit 255) via `pkill -f Ldecl`. Always make remote patterns self-avoiding.

`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` at
startup is a harmless protobuf/tensorboard warning. It appears a handful of
times and the run trains normally.

## Resuming a dead run

The launchers take a checkpoint and restore optimizer and scheduler state, so a
crashed job is resumed by pointing it at the newest checkpoint. If
`--checkpoint` is silently ignored the run restarts from scratch and numbers its
epochs from 1, so the resume must be verified rather than assumed.

A resumed run writes to a **new timestamped directory** under the same
`log_dir`. Do not verify by globbing `*/*/epoch-*.pt` and taking the highest
number: a from-scratch `epoch-5.pt` in the new directory is hidden behind the
old directory's `epoch-205.pt` under a numeric sort. Use `./verify_resume.sh`,
which reads the step numbers out of the new run's TensorBoard events — it needs
one epoch rather than five, and it checks every live job by default:

```sh
ssh ubuntu@132.145.197.191 'cd ~/CaR-constraint && ./verify_resume.sh'
ssh ubuntu@132.145.197.191 'cd ~/CaR-constraint && ./verify_resume.sh results/car1000/train_attr'
```

Steps continuing from the resume point mean the checkpoint loaded; steps
starting at `1` mean it did not. The equivalent by hand:

```sh
python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, sys
f = sorted(glob.glob(sys.argv[1]))[-1]
a = EventAccumulator(f); a.Reload()
t = a.Tags()['scalars'][0]
print(t, [e.step for e in a.Scalars(t)][:12])" \
  'results/car1000/train_interface/*/events*' 2>/dev/null | tail -1
```

(`206, 207, 208, ...` after a resume from 205 is what a healthy resume looks
like.)

```sh
# POMO, M1 (use interface + the M2 path on the other machine)
ssh ubuntu@132.145.182.17 'cd ~/CaR-constraint &&
  C=$(ls -t results/pomo1000/*/*/epoch-*.pt | head -1) &&
  nohup bash resume1000.sh attr "$C" > resume1000.log 2>&1 < /dev/null &'

# CaR, either machine -- car1000.sh resolves its own checkpoint
ssh ubuntu@132.145.182.17 'cd ~/CaR-constraint &&
  nohup bash car1000.sh > car1000.log 2>&1 < /dev/null &'
```

`car1000.sh` reads `results/composition/train_{attr,interface}/epoch-50.pt`,
which are **symlinks** into the original run directories
(`results/rerun/...` on M1, `results/matched/...` on M2). To resume from a later
epoch instead, edit its `CKPT=` line.

`ssh ... &` under a `timeout` returns exit 124 even though the launch succeeded.
Check for the process rather than trusting the exit code.

Both jobs on one A100 use ~20 GB of 40 GB, so a third job would fit on memory
but would slow everything down. A job running alone uses ~14 GB and holds the
GPU at only ~24%: this workload is latency-bound on small batches, not
GPU-bound. Do not spend the spare capacity by raising `--train_batch_size` —
the recipe must stay identical to the attr arm on M1.

## Moving a run to another machine

The whole procedure, as used to move CaR interface from M2 to M3 at epoch 150.
It relies on agent forwarding so the two remotes talk directly; the local
machine has no `ssh-agent` running by default, and **shell state does not
persist between commands**, so the agent must be started in the same command
that uses it.

```sh
eval "$(ssh-agent -s)"; ssh-add ~/.ssh/id_ed25519
ssh -A ubuntu@SOURCE '
  rsync -az --exclude "results/" --exclude "wandb/" --exclude "__pycache__/" \
        --exclude "*.log" -e "ssh -o StrictHostKeyChecking=no" \
        CaR-constraint/ ubuntu@DEST:CaR-constraint/
  cd ~/CaR-constraint
  rsync -az --relative -e "ssh -o StrictHostKeyChecking=no" \
        ./results/car1000/train_interface/TIMESTAMP/epoch-NNN.pt \
        ubuntu@DEST:CaR-constraint/'
```

Excluding `results/` keeps the transfer to ~300 MB (`data/` is 282 MB of it);
`--relative` with the `./` marker recreates the checkpoint's subpath on the
destination so the launcher and `status.sh` still find it.

A fresh box has the same base image but is missing three imports: `tqdm`,
`tensorboard-logger` and `wandb`. `wandb` is imported at the top of `train.py`
unconditionally, so it is needed even with `--wandb_logger False`. Pin `wandb`
to the source machine's version.

Then verify before killing anything on the source:

1. `md5sum` the checkpoint and `train.py`, `Trainer.py`, plus a tree hash of
   `models/ envs/ utils.py`, on both machines.
2. `diff` the `python3 train.py ...` block of the new launcher against the old
   one — every flag must match.
3. Start the new run and wait for its **first** checkpoint. It must be
   `epoch-<resume+5>`, not `epoch-5`. Only then stop the source job.

The new run writes to a new timestamped directory under the same
`results/car1000/train_interface/`, so epoch numbering stays continuous and
`status.sh` globs both directories.

## Evaluating

Two sweeps, both over the 8-variant composition grid at 128 instances.

```sh
# the grid, plus the unseen draft-limit probe -- edit --out to the run dir
ssh ubuntu@132.145.182.17 'cd ~/CaR-constraint && nohup bash grid100.sh > grid100.log 2>&1 < /dev/null &'

# the declared-bound probe: the duration row stays PUBLISHED while its bound is
# pushed out of reach, so the variant differs from the dropped one in the
# presence bit alone
ssh ubuntu@132.145.182.17 'cd ~/CaR-constraint &&
  C=$(ls results/pomo1000/*/*/epoch-XXX.pt) &&
  nohup bash probe100.sh pomo_soft_attr "$C" > probe100.log 2>&1 < /dev/null &'
```

**`matched_eval.sh` assumes one machine holds both solvers.** That is still true
for attr on M1, but the interface arm is now split: POMO interface checkpoints
are on M2 and CaR interface checkpoints past epoch 150 are on M3. To evaluate an
interface pair, first bring the two together — copying the POMO checkpoint to M3
is preferable, since M3 is idle and M2 is still training:

```sh
eval "$(ssh-agent -s)"; ssh-add ~/.ssh/id_ed25519
ssh -A ubuntu@132.145.197.191 'cd ~/CaR-constraint && rsync -az --relative \
  -e "ssh -o StrictHostKeyChecking=no" \
  ./results/pomo1000/train_pomo_soft_interface/*/epoch-NNN.pt \
  ubuntu@129.158.251.32:CaR-constraint/'
ssh ubuntu@129.158.251.32 'cd ~/CaR-constraint && ./matched_eval.sh interface NNN MMM'
```

M3 has the full `data/` tree, so `eval_composition.py` runs there unchanged.
Checkpoints at epoch <= 150 for CaR interface also still exist on M2.

`evaluate_all.py --out` is resolved **relative to the repo root and must include
the `results/` prefix** (`--out results/pomo1000`, not `--out pomo1000`). Given a
path that matches nothing it prints nothing and exits 0, which looks like a
successful no-op.

Pull and merge results into `results/composition/` locally:

```sh
cd baselines/CaR-constraint/results/composition
scp ubuntu@132.145.182.17:'~/CaR-constraint/results/composition/final100_zero.csv' m1.csv
scp ubuntu@132.145.197.191:'~/CaR-constraint/results/composition/final100_zero.csv' m2.csv
head -1 m1.csv > final100_zero.csv && tail -n +2 -q m1.csv m2.csv >> final100_zero.csv && rm m1.csv m2.csv
```

Never use `rsync --delete-excluded` against these machines — it has destroyed
data here before. Plain `rsync -az` with excludes only.

## Protocol gotchas

`tables/interface_solvers_full.tex` panel (a) is sourced from **two protocols**,
one per variant: CVRP, VRPB, VRPTW and VRPBTW from `probe_decl*.csv`
(declared-bound), and VRPL, VRPBL, VRPLTW, VRPBLTW from `final*_zero.csv`.
Regenerating it from a single CSV silently changes four numbers — at epoch 100
it turns VRPTW from -0.6% into +16.1%. Sanity check: the **attr** arm must come
out bit-identical across both protocols.

`eval_composition.py` forces `--test_opt_path /nonexistent` for every variant
except VRPBLTW, so the `Gap` column in the CSVs is 0 everywhere else. Only
VRPBLTW has a real gap, and it is read from the log line above.

The grid must not be read **across** the backhaul axis: a backhaul customer has
negative demand and refills capacity, so backhaul-free variants score worse.
Within-variant arm-versus-arm comparison is unaffected.

## Prior checkpoints

| where | what |
|---|---|
| `results/composition/train_{attr,interface}/epoch-50.pt` | CaR epoch 50 (symlinks) |
| `results/composition/train_pomo_soft_{attr,interface}/epoch-50.pt` | POMO epoch 50 |
| `results/pomo100/train_pomo_soft_{attr,interface}/*/epoch-100.pt` | POMO epoch 100 |
| `published_kopt50.pt` (M1) | the released CaR-kopt_50 model |

Local result CSVs live in `results/composition/`: `final50_{zero,draft}.csv`,
`final100_{zero,draft}.csv`, `probe_decl.csv` (epoch 50), `probe_decl100.csv`
(epoch 100), `published_car_kopt50.csv`.

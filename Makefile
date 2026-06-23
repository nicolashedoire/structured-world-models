PY ?= python3

.PHONY: all compile canon quality planner agents scale stress viz results clean

## run everything (compile + all benches)
all: compile canon quality planner agents scale stress

## byte-compile every script (fast reproducibility check)
compile:
	$(PY) -m py_compile *.py

## the headline result: random 0/3 · structured 3/3 · oracle 3/3
canon:
	$(PY) agent_canon.py

## bench 2 — open-loop model quality (one-step R² vs rollout@10/20/50)
quality:
	$(PY) model_quality.py

## bench 1 — oracle-first planner validation
planner:
	$(PY) planner_cem_solution.py

## bench 3 — closed-loop agents, auto verdict (slow)
agents:
	$(PY) agent_cem_benchmark.py

## the scaling + out-of-library experiments
scale:
	$(PY) world_scale.py
	$(PY) world_scale2.py
	$(PY) world_scale_solution.py
	$(PY) world_scale_robust.py
	$(PY) world_residual.py

## the closed-loop stress test (regime boundary)
stress:
	$(PY) closed_loop_stress.py

## regenerate the trajectory figure
viz:
	$(PY) viz_trajectories.py

## freeze all script outputs into results/
results:
	mkdir -p results
	$(PY) agent_canon.py          | tee results/agent_canon.txt
	$(PY) model_quality.py        | tee results/model_quality.txt
	$(PY) planner_cem_solution.py | tee results/planner_cem_solution.txt
	$(PY) world_scale.py          | tee results/world_scale.txt
	$(PY) world_scale2.py         | tee results/world_scale2.txt
	$(PY) world_scale_solution.py | tee results/world_scale_solution.txt
	$(PY) world_scale_robust.py   | tee results/world_scale_robust.txt
	$(PY) world_residual.py       | tee results/world_residual.txt
	$(PY) closed_loop_stress.py   | tee results/closed_loop_stress.txt
	$(PY) agent_cem_benchmark.py  | tee results/agent_cem_benchmark.txt

clean:
	rm -rf __pycache__

"""Where does the interface arm's extra training time go?

Measured overhead at the study's settings (batch 16, pomo 50, n=50): the
interface arm runs ~1.6x the attr arm's per-epoch wall time.  This isolates the
two candidate sources -- the env building the per-candidate consequence tensor
every decoding step, and the valuation network reading it -- and breaks the
second one down far enough to act on.

Run it where nothing else holds the GPU.  Shapes default to the study's.
"""

import argparse
import time

import torch

from models.consequence import CONSEQUENCE_DIM, ConsequenceValuation


def _bench(fn, repeats, warmup, device):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1e3  # ms per call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--pomo", type=int, default=50)
    parser.add_argument("--nodes", type=int, default=51)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--steps", type=int, default=51,
                        help="decoding steps per rollout, for the scale-up")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--amp", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--compile", type=str, default="on", choices=["on", "off"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (args.batch, args.pomo, args.nodes, args.rows, CONSEQUENCE_DIM)
    consequence = torch.rand(shape, device=device)
    mask = torch.rand(args.batch, args.pomo, args.nodes, device=device) > 0.3
    rows_live = consequence.shape[0] * consequence.shape[1] * consequence.shape[2] * args.rows

    print("shape {}  ->  {:,} (candidate,row) pairs per decoding step"
          .format(tuple(shape), rows_live))
    print("device {}   autocast {}\n".format(device, args.amp))

    autocast = (torch.amp.autocast("cuda", enabled=args.amp == "on")
                if device.type == "cuda"
                else torch.amp.autocast("cpu", enabled=False))

    results = {}
    for norm in ("layer", "none"):
        model = ConsequenceValuation(
            context_dim=3, hidden_dim=args.hidden, use_margin=True,
            logit_clipping=10.0, couple_rows=True, norm=norm).to(device)
        model.train()

        def whole():
            with autocast:
                model.evaluate(consequence, mask)

        results[norm] = _bench(whole, args.repeats, 5, device)

        # Training pays the backward too, and it is the larger half.
        def whole_backward():
            with autocast:
                context, value = model.evaluate(consequence, mask)
            (context.sum() + value.sum()).backward()
            model.zero_grad(set_to_none=True)

        backward_ms = _bench(whole_backward, max(5, args.repeats // 3), 3, device)
        print("--- fwd+bwd = {:.2f} ms/step ({:.2f}x forward alone) ---"
              .format(backward_ms, backward_ms / results[norm]))

        # The valuation is ~10 small ops over a (b, pomo, rows, hidden) tensor
        # plus a few over the full candidate tensor: launch-bound, not
        # FLOP-bound, which is exactly what fusion is for.
        if args.compile == "on":
            try:
                compiled = torch.compile(model, dynamic=False)

                def compiled_backward():
                    with autocast:
                        context, value = compiled.evaluate(consequence, mask)
                    (context.sum() + value.sum()).backward()
                    compiled.zero_grad(set_to_none=True)

                fused = _bench(compiled_backward, max(5, args.repeats // 3), 6, device)
                print("--- compiled fwd+bwd = {:.2f} ms/step  ({:.2f}x speedup) ---"
                      .format(fused, backward_ms / fused))
            except Exception as error:
                print("--- torch.compile failed ({}: {}) ---"
                      .format(type(error).__name__, str(error)[:120]))

        # Sub-parts of the real call graph, same inputs, so the pieces are
        # comparable against the whole.
        from models.consequence import _unit_scale
        read = model._read(consequence)
        projected = model.row_proj(read)
        active_rows = read[..., 0]
        tokens, state, active = model._row_tokens(consequence, mask)
        hidden_rows = model.row(_unit_scale(projected, norm))
        multipliers, _ = model._multipliers_from(tokens, state, active)

        parts = (
            ("_build tokens", lambda: model._row_tokens(consequence, mask)),
            ("  row_proj", lambda: model.row_proj(read)),
            ("  unit_scale", lambda: _unit_scale(projected, norm)),
            ("  row mlp", lambda: model.row(projected)),
            ("multipliers", lambda: model._multipliers_from(tokens, state, active)),
            ("context rho", lambda: model._reduce(tokens, active, model.context)),
            ("value rho", lambda: model._reduce(hidden_rows, active_rows, model.value)),
            ("_value total", lambda: model._value(consequence, multipliers)),
        )
        print("--- consequence_norm = {} ---".format(norm))
        for name, fn in parts:
            def wrapped(fn=fn):
                with autocast:
                    fn()
            try:
                ms = _bench(wrapped, args.repeats, 3, device)
                print("  {:<15} {:7.3f} ms  ({:5.1f}% of evaluate)"
                      .format(name, ms, 100 * ms / results[norm]))
            except Exception as error:
                print("  {:<15} n/a ({}: {})".format(name, type(error).__name__, error))
        print("  {:<15} {:7.3f} ms   -> {:6.2f} s per {}-step rollout\n"
              .format("EVALUATE", results[norm],
                      results[norm] * args.steps / 1e3, args.steps))

    # --- env side: does building the tensor cost as much as reading it? ------
    # Same instances, same rollout, the interface switched off and on. This is
    # the half the model profile above cannot see.
    try:
        from envs.VRPBLTWEnv import VRPBLTWEnv as Env

        def rollout(interface):
            env = Env(problem_size=args.nodes - 1, pomo_size=args.pomo,
                      device=device,
                      active_constraints=["backhaul", "route_limit", "time_window"],
                      consequence_interface=interface)
            data = env.load_dataset("data/CVRPBLTW/vrpbltw50_uniform.pkl",
                                    num_samples=args.batch)
            data = tuple(t.to(device) for t in data)
            env.load_problems(args.batch, args.pomo, problems=data)
            env.reset()
            state, _, done = env.pre_step()
            steps = 0
            while not done and steps < args.steps:
                live = state.ninf_mask == 0
                # Deterministic, so both arms take the SAME path and do the same
                # amount of work. A sampled rollout diverges between the two and
                # the difference then measures route length, not the interface.
                current = env.current_coord
                distance = ((env.depot_node_xy[:, None, :, :]
                             - current[:, :, None, :]) ** 2).sum(-1)
                choice = distance.masked_fill(~live, 1e9).argmin(-1)
                state, _, done, _ = env.step(choice, soft_constrained=True)
                steps += 1
            return steps

        print("--- env: rollout of {} decoding steps ---".format(args.steps))
        off = _bench(lambda: rollout(False), max(8, args.repeats // 2), 3, device)
        on = _bench(lambda: rollout(True), max(8, args.repeats // 2), 3, device)
        print("  interface off  {:8.1f} ms".format(off))
        print("  interface on   {:8.1f} ms   (+{:.1f}%, {:+.1f} ms)\n"
              .format(on, 100 * (on - off) / off, on - off))
    except Exception as error:
        print("--- env benchmark skipped ({}: {}) ---\n"
              .format(type(error).__name__, error))

    speedup = results["layer"] / results["none"] if results["none"] else 0
    print("dropping the layer_norm: {:.2f}x on the valuation "
          "({:.2f} -> {:.2f} ms/step)"
          .format(speedup, results["layer"], results["none"]))


if __name__ == "__main__":
    main()

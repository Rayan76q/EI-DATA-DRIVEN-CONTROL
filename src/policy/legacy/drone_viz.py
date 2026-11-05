"""Drawing the rocket, and flying it by hand (TD 4).

    from lib import rocket_viz as viz
    viz.show(states, actions)            # one picture: the flight and its trail
    viz.animate(states, actions)         # the same flight, as a video in the notebook
    viz.fly(env)                         # you are the policy: keyboard, real time, in a window
    viz.play(env)                        # the same with buttons, one click = one step (Colab)
    viz.plot_learning_curve("logs/")     # returns seen during training (Monitor files)
"""
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from matplotlib.transforms import Affine2D
from IPython.display import HTML, display

from . import rocket as R

# Colours of the scene
_SKY, _GROUND, _GROUND_EDGE = "#0b1026", "#8a8d99", "#5d606b"
_PAD, _SAFE, _UNSAFE = "#f4c542", "#4cd17a", "#ff5c5c"
_Y_TOP = 14.0
_FEET = 0.85   # the state's y is the height of the feet: the drawing is lifted by this much

# The rocket, in its own frame (m): centre of mass at (0, 0), nose up
_BODY = [(-0.22, -0.65), (0.22, -0.65), (0.22, 0.45), (-0.22, 0.45)]
_NOSE = [(-0.22, 0.45), (0.22, 0.45), (0.0, 0.95)]
_FIN_L = [(-0.22, -0.65), (-0.45, -0.85), (-0.22, -0.25)]
_FIN_R = [(0.22, -0.65), (0.45, -0.85), (0.22, -0.25)]

_rng_stars = np.random.default_rng(7)
_STARS = np.column_stack([_rng_stars.uniform(-R.X_MAX, R.X_MAX, 90),
                          _rng_stars.uniform(1.5, _Y_TOP, 90),
                          _rng_stars.uniform(0.3, 2.5, 90)])


def _decode(a):
    """(throttle, torque) of a discrete or continuous action."""
    if a is None:
        return 0.0, 0.0
    if np.ndim(a) == 0:
        return R.ACTIONS[int(a)]
    return (float(a[0]) + 1) / 2, float(a[1])


def _scene(ax):
    ax.set_facecolor(_SKY)
    ax.set_xlim(-R.X_MAX, R.X_MAX)
    ax.set_ylim(-1.2, _Y_TOP)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.scatter(_STARS[:, 0], _STARS[:, 1], s=_STARS[:, 2], c="white", alpha=0.7, lw=0)
    # Moon ground: gentle hills, flat at the pad
    xs = np.linspace(-R.X_MAX, R.X_MAX, 200)
    hills = 0.35 * np.sin(0.9 * xs) + 0.25 * np.sin(2.3 * xs + 1)
    ground = np.where(np.abs(xs) < R.PAD + 0.6, 0.0, np.clip(hills, -0.2, 0.6) * (np.abs(xs) - R.PAD) / 4)
    ax.fill_between(xs, -1.2, ground, color=_GROUND, ec=_GROUND_EDGE, lw=1.5, zorder=1)
    ax.plot([-R.PAD, R.PAD], [0, 0], color=_PAD, lw=5, solid_capstyle="butt", zorder=2)
    for xf, col in [(-R.PAD, _UNSAFE), (R.PAD, _SAFE)]:
        ax.plot([xf, xf], [0, 1.0], color="0.85", lw=1.2, zorder=2)
        ax.add_patch(Polygon([(xf, 1.0), (xf + 0.6 * np.sign(xf), 0.85), (xf, 0.7)],
                             color=col, zorder=2))


def _rocket(ax, x, y, theta, throttle=0.0, torque=0.0, alpha=1.0, rng=None):
    """Draws the rocket at (x, y), tilted by theta, with its flames."""
    tr = Affine2D().rotate(theta).translate(x, y) + ax.transData
    z = 5
    if throttle > 0.02:   # main flame, flickering
        k = rng.uniform(0.8, 1.2) if rng is not None else 1.0
        L = 0.35 + 0.9 * throttle * k
        ax.add_patch(Polygon([(-0.17, -0.65), (0.17, -0.65), (0, -0.65 - L)],
                             color="#ff8c1a", alpha=alpha, transform=tr, zorder=z - 1))
        ax.add_patch(Polygon([(-0.09, -0.65), (0.09, -0.65), (0, -0.65 - 0.6 * L)],
                             color="#ffe14d", alpha=alpha, transform=tr, zorder=z - 1))
    if abs(torque) > 0.02:   # side thruster puff near the nose
        s = -np.sign(torque)   # rotating left = puff on the right
        ax.add_patch(Polygon([(0.22 * s, 0.3), (0.22 * s, 0.5), ((0.22 + 0.4 * abs(torque)) * s, 0.4)],
                             color="#cfe8ff", alpha=alpha, transform=tr, zorder=z - 1))
    ax.add_patch(Polygon(_FIN_L, color="#d64545", alpha=alpha, transform=tr, zorder=z))
    ax.add_patch(Polygon(_FIN_R, color="#d64545", alpha=alpha, transform=tr, zorder=z))
    ax.add_patch(Polygon(_BODY, fc="#f2f2f2", ec="#9aa0aa", lw=0.8, alpha=alpha, transform=tr, zorder=z))
    ax.add_patch(Polygon(_NOSE, color="#d64545", alpha=alpha, transform=tr, zorder=z))
    ax.add_patch(Circle((0, 0.15), 0.1, fc="#4aa3ff", ec="#2b5f99", lw=0.8, alpha=alpha,
                        transform=tr, zorder=z + 1))


def _hud(ax, state, k, total=None):
    """Instruments, in green when within the soft-landing limits."""
    x, y, vx, vy, theta, _ = state
    def col(ok):
        return _SAFE if ok else _UNSAFE
    lines = [(f"t      {k * R.PARAMS.dt:5.1f} s", "white"),
             (f"alt    {max(y, 0):5.1f} m", "white"),
             (f"vx    {vx:+5.1f} m/s", col(abs(vx) < R.SAFE_VX)),
             (f"vy    {vy:+5.1f} m/s", col(abs(vy) < R.SAFE_VY)),
             (f"tilt  {np.degrees(theta):+5.0f} °", col(abs(theta) < R.SAFE_THETA))]
    if total is not None:
        lines.append((f"return {total:+6.1f}", "white"))
    h = 0.052 * len(lines) + 0.02
    ax.add_patch(FancyBboxPatch((0.01, 0.985 - h), 0.2, h - 0.005, boxstyle="round,pad=0.005",
                                transform=ax.transAxes, fc="black", ec="0.45", lw=0.8,
                                alpha=0.5, zorder=9))
    for i, (txt, c) in enumerate(lines):
        ax.text(0.02, 0.965 - 0.052 * i, txt, transform=ax.transAxes, va="top",
                family="monospace", fontsize=9, color=c, zorder=10)


def _banner(ax, outcome):
    if outcome == "landed":
        txt, c = "LANDED", _SAFE
    elif outcome in ("crashed", "lost"):
        txt, c = outcome.upper(), _UNSAFE
    else:
        return
    ax.text(0.5, 0.72, txt, transform=ax.transAxes, ha="center", va="center", fontsize=26,
            fontweight="bold", color=c, zorder=10,
            bbox=dict(boxstyle="round,pad=0.4", fc=_SKY, ec=c, lw=2, alpha=0.85))


def _explosion(ax, x, rng):
    for _ in range(40):
        r, a = rng.uniform(0, 1.6), rng.uniform(0, np.pi)
        ax.add_patch(Circle((x + r * np.cos(a), r * np.sin(a) * 0.8), rng.uniform(0.08, 0.3),
                            color=rng.choice(["#ff8c1a", "#ffe14d", "#ff5c5c", "#888888"]),
                            alpha=0.85, zorder=6))


def _outcome(state):
    """Same rule as the environment."""
    x, y, vx, vy, theta, _ = state
    if abs(x) > R.X_MAX or y > R.Y_MAX:
        return "lost"
    if y > 0:
        return "flying"
    soft = abs(vy) < R.SAFE_VY and abs(vx) < R.SAFE_VX and abs(theta) < R.SAFE_THETA
    return "landed" if soft and abs(x) < R.PAD else "crashed"


def _frame(ax, states, actions, k, total=None, rng=None):
    """Draws step k of a flight: scene, fading trail, rocket, instruments."""
    ax.clear()
    _scene(ax)
    trail = states[max(0, k - 60): k + 1]
    if len(trail) > 1:
        n = len(trail)
        for i in range(n - 1):
            ax.plot(trail[i:i + 2, 0], trail[i:i + 2, 1] + _FEET, color="#7fb3ff",
                    alpha=0.15 + 0.6 * i / n, lw=1.5, zorder=3)
    a = actions[k] if actions is not None and k < len(actions) else None
    throttle, torque = _decode(a)
    x, y, vx, vy, theta, _ = states[k]
    status = _outcome(states[k])
    if status == "crashed":
        _explosion(ax, x, rng if rng is not None else np.random.default_rng(k))
    else:
        _rocket(ax, x, max(y, 0) + _FEET, theta, throttle, torque, rng=rng)
    _hud(ax, states[k], k, total)
    if k == len(states) - 1:
        _banner(ax, status)


def show(states, actions=None, spacing=2.0, title=None):
    """One picture of a flight: its trail, and a ghost of the rocket every `spacing` metres."""
    states = np.asarray(states)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(0, 0, 1, 1)
    _scene(ax)
    ax.plot(states[:, 0], states[:, 1] + _FEET, color="#7fb3ff", lw=1.5, alpha=0.8, zorder=3)
    last = states[-1]
    shown = [last[:2]]
    for k in range(len(states) - 2, -1, -1):   # from the end, so that ghosts never hide the last rocket
        if min(np.hypot(*(states[k, :2] - p)) for p in shown) < spacing:
            continue
        shown.append(states[k, :2])
        throttle, torque = _decode(actions[k] if actions is not None and k < len(actions) else None)
        _rocket(ax, states[k, 0], states[k, 1] + _FEET, states[k, 4], throttle, torque,
                alpha=0.2 + 0.6 * k / len(states), rng=np.random.default_rng(k))
    if _outcome(last) != "crashed":
        _rocket(ax, last[0], max(last[1], 0) + _FEET, last[4])
    else:
        _explosion(ax, last[0], np.random.default_rng(0))
    _hud(ax, last, len(states) - 1)
    _banner(ax, _outcome(last))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    plt.show()


plot_trajectory = show   # older name


def animate(states, actions=None, fps=20, max_frames=200):
    """Video of a flight (HTML, plays in Jupyter and Colab)."""
    states = np.asarray(states)
    step = max(1, int(np.ceil(len(states) / max_frames)))
    frames = list(range(0, len(states), step))
    if frames[-1] != len(states) - 1:
        frames.append(len(states) - 1)
    frames += [len(states) - 1] * fps   # hold the last image for one second
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(0, 0, 1, 1)
    rng = np.random.default_rng(0)
    anim = animation.FuncAnimation(fig, lambda k: _frame(ax, states, actions, k, rng=rng),
                                   frames=frames, interval=1000 / fps * step)
    plt.close(fig)
    return HTML(anim.to_jshtml(default_mode="once"))


def play(env, seed=None):
    """You are the policy: one button per action, one click = one step (0.1 s)."""
    import ipywidgets as w

    out = w.Output()
    st = {"states": [], "actions": [], "total": 0.0, "done": False}
    base = env.unwrapped

    def show_now():
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(9, 4.6))
            fig.subplots_adjust(0, 0, 1, 1)
            states = np.array(st["states"])
            k = len(states) - 1
            _frame(ax, states, st["actions"] + [None], k, total=st["total"],
                   rng=np.random.default_rng(k))
            plt.show()

    def reset(_=None):
        env.reset(seed=seed)
        st.update(states=[base.state.copy()], actions=[], total=0.0, done=False)
        show_now()

    def act(a, repeat=1):
        def click(_):
            for _ in range(repeat):
                if st["done"]:
                    return
                _, r, term, trunc, info = env.step(a)
                st["total"] += r
                st["actions"].append(a)
                st["states"].append(base.state.copy())
                st["done"] = term or trunc
            show_now()
        return click

    buttons = []
    for a, (name, icon) in enumerate(zip(R.ACTION_NAMES, ["pause", "fire", "rotate-left", "rotate-right"])):
        b = w.Button(description=name, icon=icon, layout=w.Layout(width="130px"))
        b.on_click(act(a))
        buttons.append(b)
    wait5 = w.Button(description="nothing x5", icon="forward", layout=w.Layout(width="130px"))
    wait5.on_click(act(0, repeat=5))
    again = w.Button(description="restart", icon="refresh", button_style="warning",
                     layout=w.Layout(width="110px"))
    again.on_click(reset)
    display(w.HBox(buttons + [wait5, again]), out)
    reset()


# Keys of fly(), by priority: one discrete action per step
_KEYS = {"up": 1, "left": 2, "right": 3}


class _Pilot:
    """State of a flight by hand, in real time. Separate from the window, so that it can be tested."""

    def __init__(self, env, ax, seed=None):
        self.env, self.base, self.ax, self.seed = env, env.unwrapped, ax, seed
        self.held, self.pressed = set(), set()
        self.reset()

    def reset(self):
        self.env.reset(seed=self.seed)
        self.states, self.actions = [self.base.state.copy()], []
        self.total, self.done, self.started, self.outcome = 0.0, False, False, "flying"
        self.draw()

    def action(self):
        """Main engine first, then rotations. A key pressed since the last step counts as held:
        keyboard auto-repeat sends release/press pairs, and a step must not fall in between."""
        keys = self.held | self.pressed
        self.pressed = set()
        return next((a for k, a in _KEYS.items() if k in keys), 0)

    def on_press(self, key):
        if key in _KEYS:
            self.held.add(key)
            self.pressed.add(key)
            self.started = True
        elif key == "r":
            self.reset()

    def on_release(self, key):
        self.held.discard(key)

    def tick(self):
        """One time step (0.1 s), if the flight is on."""
        if not self.started or self.done:
            return
        a = self.action()
        _, r, term, trunc, info = self.env.step(a)
        self.total += r
        self.outcome = "time limit" if trunc else info["outcome"]
        self.actions.append(a)
        self.states.append(self.base.state.copy())
        self.done = term or trunc
        self.draw()

    def draw(self):
        states = np.array(self.states)
        k = len(states) - 1
        last = self.actions[-1:] if not self.done else [None]
        _frame(self.ax, states, self.actions + last, k, total=self.total,
               rng=np.random.default_rng(k))
        if not self.started:
            self.ax.text(0.5, 0.55, "↑ main engine    ← → rotate\n\npress a key to start",
                         transform=self.ax.transAxes, ha="center", va="center", fontsize=14,
                         color="white", zorder=10)
        elif self.done:
            self.ax.text(0.5, 0.55, "R: try again    Q: back to the notebook",
                         transform=self.ax.transAxes, ha="center", va="center", fontsize=12,
                         color="white", zorder=10)
        self.ax.figure.canvas.draw_idle()


def fly(env, seed=None, speed=1.0):
    """You are the policy, in real time, in a separate window.

    Keys: up = main engine, left / right = rotate, R = restart, Q (or close the window) = back to
    the notebook. The notebook waits until the window is closed. With the same `seed`, every
    restart starts from the same state. Where no window can open (Colab, a server), falls back
    to `play` (buttons in the notebook).
    """
    import matplotlib
    previous = matplotlib.get_backend()
    for backend in ("TkAgg", "MacOSX", "QtAgg"):
        try:
            plt.switch_backend(backend)
            break
        except Exception:
            continue
    else:
        print("No window can open here (Colab?): buttons in the notebook instead.")
        return play(env, seed=seed)
    try:
        with plt.rc_context({"toolbar": "None"}):
            fig, ax = plt.subplots(figsize=(10, 5.2))
        fig.subplots_adjust(0, 0, 1, 1)
        fig.canvas.manager.set_window_title("Rocket: arrows to fly, R to restart, Q to quit")
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)   # no matplotlib shortcuts
        pilot = _Pilot(env, ax, seed=seed)

        def press(event):
            if event.key in ("q", "escape"):
                plt.close(fig)
            else:
                pilot.on_press(event.key)

        fig.canvas.mpl_connect("key_press_event", press)
        fig.canvas.mpl_connect("key_release_event", lambda event: pilot.on_release(event.key))
        timer = fig.canvas.new_timer(interval=int(1000 * R.PARAMS.dt / speed))
        timer.add_callback(pilot.tick)
        timer.start()
        plt.show(block=True)
        timer.stop()
        print(f"{len(pilot.actions)} steps, return {pilot.total:+.1f}, outcome: {pilot.outcome}")
    finally:
        plt.switch_backend(previous)


def plot_learning_curve(log_dirs, window=50):
    """Episode returns logged by stable_baselines3's Monitor, smoothed.

    `log_dirs`: a folder, or {label: folder} to compare several runs.
    """
    if isinstance(log_dirs, str):
        log_dirs = {os.path.basename(log_dirs.rstrip("/")): log_dirs}
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, folder in log_dirs.items():
        # one file per parallel environment (PPO: 8): merge the episodes in the order
        # they ended (column t), so that the x axis counts the steps of all environments
        # Monitor(env, "logs/dqn/") writes the hidden file logs/dqn/.monitor.csv, that * skips
        files = glob.glob(os.path.join(folder, "*monitor.csv")) + glob.glob(os.path.join(folder, ".monitor.csv"))
        rows = [np.loadtxt(f, delimiter=",", skiprows=2, ndmin=2) for f in files]   # r, l, t
        rows = [d for d in rows if len(d)]
        if not rows:
            continue
        data = np.concatenate(rows)
        data = data[np.argsort(data[:, 2], kind="stable")]
        returns, steps = data[:, 0], np.cumsum(data[:, 1])
        smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], smooth, label=label)
    ax.axhline(100, color="tab:green", ls="--", lw=1, label="soft landing (+100)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"episode return (mean of {window})")
    ax.legend()
    plt.show()

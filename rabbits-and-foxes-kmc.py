# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: chme5137
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rabbits and foxes, by kinetic Monte Carlo
#
# **CHME 5137, Lecture 10, Tuesday October 13, 2026.** Started in class, finished for
# homework.
#
# *Written for 2026 by Claude Code (Claude Opus 5.5) from Prof. West's KMC question of
# earlier years, and checked by Prof. West. The problem is from Chapter 1 of H. Scott
# Fogler's "Essentials of Chemical Reaction Engineering".*
#
# ---
#
# Same farm, same rate constants: 400 rabbits and 200 foxes at the start, and
#
# $$\frac{dR}{dt} = k_1 R - k_2 R F \qquad\qquad \frac{dF}{dt} = k_3 R F - k_4 F$$
#
# with $k_1 = 0.015$ day$^{-1}$, $k_2 = 0.00004$ day$^{-1}$ foxes$^{-1}$,
# $k_3 = 0.0004$ day$^{-1}$ rabbits$^{-1}$ and $k_4 = 0.04$ day$^{-1}$.
#
# In the `solve_ivp` homework you found that the foxes fall to **under two** in each
# trough. The ODEs don't mind: 1.6 foxes carry on and breed. Real foxes can't be
# fractional, and two foxes can easily both die before they meet a rabbit. So now we
# treat every rabbit and fox as an individual, and every birth and death as a random
# event. That is **kinetic Monte Carlo** (KMC).
#
# ### The events
#
# There are four events. Each has a rate, in events per day, which changes as the
# populations change:
#
# | Event | Rate | Effect |
# |---|---|---|
# | a rabbit is born | $k_1 R$ | $R \to R + 1$ |
# | a rabbit is eaten | $k_2 R F$ | $R \to R - 1$ |
# | a fox is born | $k_3 R F$ | $F \to F + 1$ |
# | a fox dies | $k_4 F$ | $F \to F - 1$ |
#
# The rates are easy to calculate, so we use the **rejection-free** algorithm, which you
# worked out in class from
# [Wikipedia's Kinetic Monte Carlo page](https://en.wikipedia.org/wiki/Kinetic_Monte_Carlo).
# Writing it down in your own words is the first part of task 1.
#
# ### How to work
#
# - **Commit as you go.** After each numbered task works, Restart & Run All, then
#   stage, commit with a message that says what you did, and push. At least one commit
#   per task. Your commit history is part of what gets looked at. **Commit the `.ipynb`
#   and its paired `.py` together**: the `.py` is the one whose diffs you can read on
#   GitHub. If saving the notebook doesn't update the `.py`, see *Setting up Jupytext* on
#   Canvas.
# - **Answer the questions in words**, in the markdown cells marked *Answer*.
#   Double-click a cell to edit it.
# - **Random numbers:** use a NumPy generator, `rng = np.random.default_rng(seed)`,
#   not the legacy `np.random.*` functions. Give it a seed, so that Restart & Run All
#   gives the same answers every time. Check what interval each function returns,
#   and what its arguments mean, before you trust it.
# - **Let bugs shout.** If a population goes negative or an event isn't one of the
#   four, `raise` an error. Don't hide it with `max(0, F)`: that turns a bug you'd
#   notice into a wrong answer you won't.

# %%
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

k1 = 0.015  # 1/day
k2 = 0.00004  # 1/day/fox
k3 = 0.0004  # 1/day/rabbit
k4 = 0.04  # 1/day

R0, F0 = 400, 200  # rabbits and foxes at t = 0
t_end = 600  # days

# %% [markdown]
# The deterministic answer, for comparison. This is the `solve_ivp` homework's
# solution, at a tight tolerance.

# %%
def dydt(t, y):
    R, F = y
    return [k1 * R - k2 * R * F, k3 * R * F - k4 * F]

ode = solve_ivp(dydt, (0, t_end), [R0, F0], rtol=1e-10, atol=1e-8, dense_output=True)
t_ode = np.linspace(0, t_end, 3001)
R_ode, F_ode = ode.sol(t_ode)

plt.plot(t_ode, R_ode, label="rabbits (ODE)")
plt.plot(t_ode, F_ode, label="foxes (ODE)")
plt.xlabel("time (days)")
plt.ylabel("population")
plt.legend()
plt.show()


# %% [markdown]
# ## 1. One trajectory
#
# **(a) The algorithm, in your own words.** Before you write any code, write the
# rejection-free algorithm as numbered steps, for rabbits and foxes: what each of
# Wikipedia's symbols ($k$, $r_{ki}$, $R_{ki}$, $Q_k$, $u$, $u'$) becomes here, how the
# next event is chosen, and how far the clock moves. Then work the **first step by
# hand**, from 400 rabbits and 200 foxes, with $u = 0.5$ and $u' = 0.3$: which event
# happens, and how many minutes pass?

# %% [markdown]
# *Answer:
#
# **Variables:**
#
# $r_1 = k_1$ $R$ $(Rabbit$ $Birth)$
#
# $r_2 = k_2$ $RF$ $(Rabbit$ $Death)$
#
# $r_3 = k_3$ $RF$ $(Fox$ $Birth)$
#
# $r_4 = k_4$ $F$ $(Fox$ $Death)$
#
# $Q_k = r_1 + r_2 + r_3 + r_4$ $(Total$ $Rate)$
#
# $u$ is a random number used to find the event $i$ that satisfies $R_{k,i-1}$ $<$ $uQ_k$ $\le R_{ki}$
#
# $u'$ is a random number used to update time. $\Delta t$ $=$ $Q_{k}^{-1}$ $ln(1/u')$
#
# **Hand Calculation**
#
# $R = 400$
#
# $F = 200$
#
# $u = 0.5$
#
# $u' = 0.3$
#
# $k_1 = 0.015$ $day ^{-1}$
#
# $k_2 = 0.00004$ $day ^{-1}$ $foxes ^{-1}$
#
# $k_3 = 0.0004$ $day ^{-1}$ $rabbits ^{-1}$
#
# $k_4 = 0.04$ $day ^{-1}$
#
# $r_1 (Rabbit$ $Birth)$ $= 0.015$ $\times$ $400 = 6$ $days ^{-1}$
#
# $r_2 (Rabbit$ $Death)$ $= 0.00004$ $\times$ $400$ $\times$ $200 = 3.2$ $days ^{-1}$
#
# $r_3 (Fox$ $Birth)$ $= 0.0004$ $\times$ $400$ $\times$ $200 = 32$ $days ^{-1}$
#
# $r_4 (Fox Death) = 0.04$ $\times$ $200 = 8$ $days ^{-1}$
#
# $Q = 6 + 3.2 + 32 + 8 = 49.2$ $days ^{-1}$
#
# $Event$ $1$ $Interval = [0, 6)$
#
# $Event$ $2$ $Interval = [6, 9.2)$
#
# $Event$ $3$ $Interval = [9.2, 41.2)$
#
# $Event$ $4$ $Interval = [41.2, 49.2)$
#
# $uQ = 0.5$ $\times$ $49.2 = 24.6$
#
# Event 3 happens as 14.76 falls between 9.2 and 41.2
#
# $\Delta t$ $=$ $Q_{k}^{-1}$ $ln(1/u')$
#
# $\Delta t$ $=$ $(1/49.2)$ $\times$ $ln(1/0.3)$
#
# $\Delta t$ $=$ $0.024$ $days$ $= 35.24$ $minutes$*

# %% [markdown]
# **(b) The function.** Write
#
# ```python
# def kmc(R0, F0, t_end, rng):
#     ...
#     return times, rabbits, foxes   # three NumPy arrays, one entry per event
# ```
#
# that runs your algorithm from $t = 0$ until the clock passes `t_end`. If the total
# rate is ever zero, nothing more can happen, so stop there.
#
# Run it once. How many events were there, and how long did it take? Plot rabbits and
# foxes against time, on top of the ODE solution. (`plt.step(times, foxes, where="post")`
# draws a population that jumps, rather than one that slopes.) Run it a few more times
# with different seeds. How does a KMC trajectory differ from the ODE solution?

# %%
def kmc(R0, F0, t_end, rng):
    t = 0
    R = R0
    F = F0
    t_1 = [t]
    R_1 = [R]
    F_1 = [F]

    while t < t_end:
        r1 = k1 * R
        r2 = k2 * R * F
        r3 = k3 * R * F
        r4 = k4 * F

        Q = r1 + r2 + r3 + r4
        if Q == 0:
            break
        u = rng.random()
        x = u * Q
        if x < r1:
            R = R + 1
        elif x < r1 + r2:
            R = R - 1
        elif x < r1 + r2 + r3:
            F = F + 1
        else:
            F = F - 1

        u_prime = rng.random()
        dt = -np.log(u_prime) / Q
        t = t + dt

        if R < 0 or F < 0:
            raise ValueError(f"Negative Population: R = {R}, F = {F}")

        t_1.append(t)
        R_1.append(R)
        F_1.append(F)

    return np.array(t_1), np.array(R_1), np.array(F_1)

rng = np.random.default_rng(42)

t_i = time.perf_counter()
times, rabbits, foxes = kmc(R0, F0, t_end, rng)
t_f = time.perf_counter()

time_taken = t_f - t_i
print(f"Time taken for KMC simulation = {time_taken:.4f} seconds")
print(f"Number of events = {len(times) - 1}")

plt.step(times, rabbits, where='post', label="Rabbits (KMC)")
plt.step(times, foxes, where='post', label="Foxes (KMC)")
plt.plot(t_ode, R_ode, "--", label="Rabbits (ODE)")
plt.plot(t_ode, F_ode, "--", label="Foxes (ODE)")
plt.xlabel("Time (Days)")
plt.ylabel("Population")
plt.legend()
plt.show()


# %% [markdown]
# *Answer: For seed 42, the simulation took 23,328 events and finished in about 0.1027 seconds. When comparing KMC to the ODE, the biggest difference is that KMC moves in discrete integer steps of plus or minus one instead of smooth curves, so the populations never have fractional animals. Because each event happens randomly, the timing and height of the peaks drift away from the ODE curve. In this run, the first fox peak was slightly higher and the second peak happened earlier and was much smaller. Testing other seeds also shows that foxes can easily die out completely when their numbers drop very low in the trough. In the ODE, foxes drop to around 1.6 and always bounce back, but in KMC, once the fox count hits zero, no more foxes can ever be born, leaving the rabbits to grow without any predators.*

# %% [markdown]
# ## 2. Many trajectories, and where they end up
#
# Plot the foxes from 20 trajectories on one graph (thin lines, `lw=0.5`), with the ODE
# solution on top.
#
# - What do the trajectories have in common, and where do they spread apart?
# - What happens to a trajectory once the foxes reach zero? Once the rabbits reach
#   zero? Why can't it ever recover?
# - Where in the cycle do the foxes usually die out, and why there? (Your answer to
#   task 5 of the `solve_ivp` homework is relevant.)

# %%
rng = np.random.default_rng(33)

for i in range(20):
    times, rabbits, foxes = kmc(R0, F0, t_end, rng)
    plt.step(times, foxes, where="post", lw=0.5)
    
plt.plot(t_ode, F_ode, "k--", lw=1, label="Foxes (ODE)")
plt.xlabel("Time (Days)")
plt.ylabel("Foxes Population")
plt.legend()
plt.show()


# %% [markdown]
# *Answer:
#
# a) All 20 trajectories closely follow the ODE during the first cycle because populations are large, keeping fluctuations to a small amount. They spread apart significantly during the deep trough days 150–300 when populations drop close to single digits, causing large stochastic variations in the timing and amplitude of the second peak.
#
# b) If the foxes reach 0 (F = 0), then the possibility of both the events fox birth and fox death happening will be 0. The rabbits will then have no threats and they grow exponentially. If the rabbits reach 0 (R = 0), then the events rabbit birth, rabbit death and fox birth will become 0. The remaining foxes will have no more food and can only die through the event fox death until both species reach 0.
#
# c) Foxes die out in the trough around days 180 - 300. In this region, rabbits are nearly depleted, so the fox birth rate drops close to zero while fox mortality continues. Because the population drops to single digits (the ODE falls to around 1.6), random deaths can easily eliminate the final remaining individuals before rabbits recover.*

# %% [markdown]
# ## 3. The second fox peak
#
# The ODE's foxes peak at about day 32, and again at about day 428, at about 2804
# foxes each time. When is the second peak in a KMC trajectory, and how high?
#
# A noisy trajectory has lots of tiny local maxima, so we need a definition. Use this
# one: **the largest number of foxes after day 200, counted as a peak only if it is
# above 100 foxes and isn't within 5 days of either end of the window** (day 200 or
# day 600). Write a function `second_peak(times, foxes)` that returns `(time, height)`,
# or `None` if there isn't one.
#
# Run 100 trajectories. In what fraction is there a second peak? For those that have
# one, give the mean and median of its time and of its height, and the interquartile
# range (25th to 75th percentile, `np.percentile`) of each. Plot histograms.
#
# How do these compare with the ODE's second peak? If they differ, why? And what does
# the definition get wrong? Find a trajectory where it picks something you wouldn't
# call "the second peak", if you can.

# %%
def second_peak(times, foxes):
    x = (205 < times ) & (times < 595)

    t_1 = times[x]
    f_1 = foxes[x]

    if len(f_1) == 0:
        return None

    y = np.argmax(f_1)
    fox_max = f_1[y]

    if fox_max > 100:
        return t_1[y], fox_max
    else:
        return None

rng = np.random.default_rng(30)
peak_time = []
peak_height = []

for i in range(100):
    times, rabbits, foxes = kmc(R0, F0, t_end, rng)
    result = second_peak(times, foxes)
    if result is not None:
        t_2, f_2 = result
        peak_time.append(t_2)
        peak_height.append(f_2)

peak_time = np.array(peak_time)
peak_height = np.array(peak_height)

print(f"Fraction with second peak: {len(peak_time) / 100:.2f}")
print(f"Peak Time: mean = {np.mean(peak_time):.1f}, median = {np.median(peak_time):.1f}, iqr = {np.percentile(peak_time, [25, 75])}")
print(f"Peak Height: mean = {np.mean(peak_height):.1f}, median = {np.median(peak_height):.1f}, iqr = {np.percentile(peak_height, [25, 75])}")

plt.figure()
plt.hist(peak_time, bins=15, edgecolor = "black")
plt.axvline(428, color="red", linestyle="--", label="ODE (428 Day)")
plt.xlabel("Peak Time (Days)")
plt.ylabel("Count")
plt.title("Distribution of Second Fox Peak Times")
plt.legend()
plt.show()

plt.figure()
plt.hist(peak_height, bins=15, edgecolor="black")
plt.axvline(2804, color="red", linestyle="--", label="ODE (2804 Foxes)")
plt.xlabel("Peak Height (Foxes)")
plt.ylabel("Count")
plt.title("Distribution of Second Fox Peak Heights")
plt.legend()
plt.show()

# %% [markdown]
# *Answer: In the ODE, the second peak occurs at day 428 with a height of 2804 foxes. In KMC, the surviving peaks arrive a bit earlier, median (~400 days) and are lower on average (median ~2283). The surviving trajectories depend on survival through the trough. If the fox population drops to a few individuals rather than the fractional ~1.6 foxes of the ODE, rabbits can grow faster without any predatory threat. A rapid rabbit explosion leads to earlier fox recovery, while variations in predation timing often cut off the peak before reaching the deterministic height.
#
# The peak definition just takes whatever the highest point is between days 205 and 595 as long as it gets over 100. I think this mostly messes up because 100 foxes isn't really that many, so sometimes a run that is almost dead just gets a tiny bump over 100 and it counts that as the second peak even though it wasn't a real comeback. Also, if the curve is still going up at the very end around day 590, it might just pick that spot because it ran out of time instead of waiting for the real peak.*

# %% [markdown]
# ## 4. How many runs are enough?
#
# Every number in task 3 came from random runs, so it has its own uncertainty. Repeat
# task 3 with 100, 300, 1000 and 3000 trajectories. (Time it first, and work out how
# long 3000 will take before you start.)
#
# - How do the mean second-peak time and height change with the number of runs?
# - Work out the **standard error** of each mean, $s / \sqrt{n}$, where $s$ is the
#   standard deviation and $n$ the number of trajectories with a peak. How does it
#   scale with $n$?
# - Give the expected time and height of the second peak, and their interquartile
#   ranges, to a precision you can justify. Say how you justified it.

# %%
runs = [100, 300, 1000, 3000]
peak_t = []
peak_h = []
rng = np.random.default_rng(42)

for i in range(1, 3001):
    times, rabbits, foxes = kmc(R0, F0, t_end, rng)
    z = second_peak(times, foxes)
    if z is not None:
        peak_t.append(z[0])
        peak_h.append(z[1])

    if i in runs:
        n = len(peak_t)

        mean_t = np.mean(peak_t)
        s_t = np.std(peak_t, ddof=1)
        se_t = s_t / np.sqrt(n)
        iqr_t = np.percentile(peak_t, [25, 75])

        mean_h = np.mean(peak_h)
        s_h = np.std(peak_h, ddof=1)
        se_h = s_h / np.sqrt(n)
        iqr_h = np.percentile(peak_h, [25, 75])

        print(f"--- Runs: {i} (n = {n} peaks detected) ---")
        print(f"Peak Time:   mean = {mean_t:.2f} ± {se_t:.2f} days,  iqr = [{iqr_t[0]:.1f}, {iqr_t[1]:.1f}]")
        print(f"Peak Height: mean = {mean_h:.1f} ± {se_h:.1f} foxes, iqr = [{iqr_h[0]:.1f}, {iqr_h[1]:.1f}]\n")

# %% [markdown]
# *Answer:
# a) When we run more trajectories from 100 up to 3000, the average values for the second peak time and peak height don't really jump all over the place, but they stay pretty much around 401 days and 2340 foxes. Doing only 100 runs has more random noise, but once you do 3000 runs it feels much smoother and stays centered.
#
# b) The standard error gets smaller when you increase the runs because of the formula. Since the number of peaks $n$ is on the bottom, having more runs divides the spread by a bigger number. When we went from about 32 peaks at 100 runs to almost 1000 peaks at 3000 runs, the error dropped down from like 4.7 days to around 1 day, and the height error dropped from around 96 foxes to about 18 foxes. So it scales down as $n$ goes up.
#
# c) Looking at the 3000-run result, we can probably write the peak time as about $401 \pm 1$ days and the peak height as around $2343 \pm 19$ foxes. The IQRs are roughly 379 to 420 days and 1939 to 2685 foxes. We can justify rounding the time to the nearest whole number because the error is around 1 day, so having two decimal places doesn't make much sense when the noise changes the day anyway. For the height, the error is around 18 foxes, so reporting it to the nearest single fox is probably too specific.*

# %% [markdown]
# ## 5. How likely is it that the foxes die out?
#
# From your largest set of runs, what fraction have **no foxes left at day 600**?
# That fraction $p$ estimates a probability. Its standard error is
# $\sqrt{p(1-p)/N}$ for $N$ runs.
#
# - Give the probability that the foxes die out before 600 days, with its uncertainty.
# - How many runs would you need to know it to within ±1 percentage point? To within
#   ±0.1?
# - Plot a histogram of the times at which the foxes die out. When does it usually
#   happen?

# %%
rng = np.random.default_rng(42)
N = 3000
ext = 0
ext_times = []

for i in range(N):
    times, rabbits, foxes = kmc(R0, F0, t_end, rng)
    if foxes[-1] == 0:
        ext += 1
        t_ext = times[foxes == 0][0]
        ext_times.append(t_ext)

p = ext / N
se_p = np.sqrt(p * (1 - p) / N)

N_1pct = p * (1 - p) / (0.01**2)
N_pt1pct = p * (1 - p) / (0.001**2)

print(f"Runs for ±1%:   {N_1pct:.0f}")
print(f"Runs for ±0.1%: {N_pt1pct:.0f}")

print(f"Total runs: {N}")
print(f"Extinct runs: {ext}")
print(f"Extinction probability (p): {p:.4f} ± {se_p:.4f} ({p:.2%} ± {se_p:.2%})")

plt.figure()
plt.hist(ext_times, bins=25, edgecolor='black')
plt.xlabel("Extinction Time (Days)")
plt.ylabel("Number of Trajectories")
plt.title("Distribution of Fox Extinction Times")
plt.show()

# %% [markdown]
# *Answer:
# a) Out of the 3000 runs, foxes died out in 2075 runs, so the extinction probability is about $0.6917 \pm 0.0084$ (around $69.2\% \pm 0.8\%$).
#
# b) To get the error within $\pm 1\%$ we would need around 2,133 runs, and to get it within $\pm 0.1\%$ it jumps all the way up to about 213,264 runs because dividing by a smaller squared decimal makes the required number huge.
#
# c) Looking at the histogram, almost all the extinctions happen between days 230 and 320, especially around day 260. That makes sense because right after the foxes eat all the rabbits, the rabbits drop close to zero, so the foxes run out of food and starve to death during that first big crash. There is also a tiny second bump near day 580 or 600 where a few runs that survived the first cycle end up crashing when the second wave goes down.*

# %% [markdown]
# ## 6. Your project
#
# One paragraph. Is there any part of your project where the numbers are small
# enough, or the events discrete and random enough, that a continuum model (an ODE or
# a PDE) could mislead you, the way the ODE does about the foxes? Think about
# molecules on a surface, cells, bubbles, particles, defects, failures. If there is,
# what would the "events" and their rates be? If there isn't, why is a continuum model
# good enough?

# %% [markdown]
# *Answer: In wafer wet cleaning, we care a lot about removing tiny leftover residue and small contaminant particles from the surface. A regular continuum model like an ODE or PDE assumes the residue is just a smooth concentration that spreads out and washes away gradually. But when the wafer is almost completely clean, we aren't dealing with a smooth concentration anymore, we only have a few isolated, discrete particles left on a whole die. An ODE might say something like "99.9% clean" or predict an average residue concentration of 0.001, but in real manufacturing, a single particle that gets stuck or fails to detach is enough to ruin the chip. A continuum model averages that out and misses the chance of having that one killer defect. To model this properly, we would need to switch from smooth rate equations to something discrete like Kinetic Monte Carlo, where we track individual particles and treat their detachment or sticking as random probabilistic events.*

# %% [markdown]
# ## Stretch problems
#
# **(a) Faster.** Time one trajectory. Then find where the time goes (`%prun`, or
# time the pieces). Try Python's `random.random()` in place of `rng.random()`,
# appending to lists versus filling a preallocated array, and recording only every
# 100th event. How much faster can you make it?
#
# **(b) Bigger, on Explorer.** The `.py` paired with this notebook is a script.
# Make a copy that runs 10,000 trajectories and saves the second-peak times and heights
# and the extinction times with `np.savetxt`, and run it as a batch job on Explorer,
# as in Lecture 9. Do your answers to tasks 4 and 5 hold up?
#
# **(c) Is the random clock worth it?** Wikipedia notes that advancing the clock by
# $\Delta t = 1/Q_k$ every step, instead of an exponentially distributed $\Delta t$, gives
# "the same *average* time scale" (and that note is marked *citation needed*). Test it.
# Make a copy of your `kmc` function that uses $\Delta t = 1/Q$, and run both versions
# many times. Compare the probability that the foxes die out by day 600, the fraction of
# runs with a second peak, and the mean time and height of that peak, each with its
# standard error. Which differences are bigger than their uncertainty? How many runs did
# you need before you could tell? (Several thousand of each; (a) and (b) help.) Why might
# some quantities change and others not?

# %%

# %% [markdown]
# ---
#
# ### When you're done
#
# Restart & Run All. Check that every cell ran, in order, with no errors. Commit and
# push. Then check on GitHub that your fork shows your latest commit, and that the
# notebook displays with its plots.

import streamlit as st
import numpy as np
import pandas as pd
import random

# ---------------------------
# Markov chain construction
# ---------------------------

def build_volleyball_model(p):
    """Build transition matrix P for a race-to-5, win-by-2 volleyball game.

    State design:
    - Sprint phase: all score pairs (a,b) with a,b in {0,1,2,3,4},
      excluding 3-3 (we jump into the duel phase there).
    - Duel phase: relative states "-1", "0", "+1" (score differential after 3-3).
    - Absorbing: "A_Wins", "B_Wins".
    """
    sprint_states = []
    for a in range(5):
        for b in range(5):
            if (a, b) == (3, 3):
                continue
            sprint_states.append(f"({a},{b})")

    duel_states = ["-1", "0", "+1"]
    absorbing_states = ["A_Wins", "B_Wins"]

    transient_states = sprint_states + duel_states
    all_states = transient_states + absorbing_states

    n_transient = len(transient_states)
    n_total = len(all_states)

    state_to_idx = {s: i for i, s in enumerate(all_states)}

    P = np.zeros((n_total, n_total))

    for state in all_states:
        i = state_to_idx[state]

        if state in absorbing_states:
            P[i, i] = 1.0
            continue

        if state in sprint_states:
            a, b = map(int, state.strip("()").split(","))

            # Team A wins rally (prob p)
            na, nb = a + 1, b
            if na >= 5 and na - nb >= 2:
                P[i, state_to_idx["A_Wins"]] += p
            elif (na, nb) == (3, 3):
                P[i, state_to_idx["0"]] += p
            elif (na, nb) == (4, 3):
                P[i, state_to_idx["+1"]] += p
            elif (na, nb) == (3, 4):
                P[i, state_to_idx["-1"]] += p
            else:
                next_state = f"({na},{nb})"
                if next_state in state_to_idx:
                    P[i, state_to_idx[next_state]] += p

            # Team B wins rally (prob 1-p)
            na, nb = a, b + 1
            if nb >= 5 and nb - na >= 2:
                P[i, state_to_idx["B_Wins"]] += (1 - p)
            elif (na, nb) == (3, 3):
                P[i, state_to_idx["0"]] += (1 - p)
            elif (na, nb) == (4, 3):
                P[i, state_to_idx["+1"]] += (1 - p)
            elif (na, nb) == (3, 4):
                P[i, state_to_idx["-1"]] += (1 - p)
            else:
                next_state = f"({na},{nb})"
                if next_state in state_to_idx:
                    P[i, state_to_idx[next_state]] += (1 - p)

        else:  # duel state
            diff = 0 if state == "0" else int(state)

            # Team A wins rally
            new_diff = diff + 1
            if new_diff >= 2:
                P[i, state_to_idx["A_Wins"]] += p
            elif new_diff == 1:
                P[i, state_to_idx["+1"]] += p
            elif new_diff == 0:
                P[i, state_to_idx["0"]] += p
            elif new_diff == -1:
                P[i, state_to_idx["-1"]] += p

            # Team B wins rally
            new_diff = diff - 1
            if new_diff <= -2:
                P[i, state_to_idx["B_Wins"]] += (1 - p)
            elif new_diff == 1:
                P[i, state_to_idx["+1"]] += (1 - p)
            elif new_diff == 0:
                P[i, state_to_idx["0"]] += (1 - p)
            elif new_diff == -1:
                P[i, state_to_idx["-1"]] += (1 - p)

    Q = P[:n_transient, :n_transient]
    R = P[:n_transient, n_transient:]

    return all_states, transient_states, absorbing_states, P, Q, R, state_to_idx

# ---------------------------
# Analytical engine
# ---------------------------

def analytical_solution(Q, R):
    """Compute F, t, B for the absorbing Markov chain."""
    n = Q.shape[0]
    I = np.eye(n)
    F = np.linalg.inv(I - Q)
    t = F.sum(axis=1)
    B = F @ R
    return F, t, B

# ---------------------------
# Simulation engine
# ---------------------------

def simulate_game(p):
    """Simulate a single volleyball game. Return absorbing state, total points, state visits."""
    current_state = "(0,0)"
    points = 0
    states_visited = {}
    max_points = 1000  # safety

    while current_state not in ["A_Wins", "B_Wins"] and points < max_points:
        states_visited[current_state] = states_visited.get(current_state, 0) + 1

        if current_state.startswith("("):
            # Sprint phase
            a, b = map(int, current_state.strip("()").split(","))
            if random.random() < p:
                a += 1
            else:
                b += 1
            points += 1

            if a >= 5 and a - b >= 2:
                current_state = "A_Wins"
            elif b >= 5 and b - a >= 2:
                current_state = "B_Wins"
            elif (a, b) == (3, 3):
                current_state = "0"
            elif (a, b) == (4, 3):
                current_state = "+1"
            elif (a, b) == (3, 4):
                current_state = "-1"
            else:
                current_state = f"({a},{b})"
        else:
            # Duel phase: track differential
            diff = 0 if current_state == "0" else int(current_state)
            if random.random() < p:
                diff += 1
            else:
                diff -= 1
            points += 1

            if diff >= 2:
                current_state = "A_Wins"
            elif diff <= -2:
                current_state = "B_Wins"
            elif diff == 1:
                current_state = "+1"
            elif diff == 0:
                current_state = "0"
            elif diff == -1:
                current_state = "-1"

    return current_state, points, states_visited

def run_simulations(p, n_trials, transient_states, absorbing_states):
    """Run Monte Carlo simulations and track game-length distribution."""
    results = {
        "A_Wins": 0,
        "B_Wins": 0,
        "total_points": 0,
        "state_visits": {s: 0 for s in transient_states},
        "length_counts": {},
    }

    for _ in range(n_trials):
        final_state, points, visits = simulate_game(p)
        if final_state == "A_Wins":
            results["A_Wins"] += 1
        elif final_state == "B_Wins":
            results["B_Wins"] += 1

        results["total_points"] += points
        results["length_counts"][points] = results["length_counts"].get(points, 0) + 1

        for s, c in visits.items():
            if s in results["state_visits"]:
                results["state_visits"][s] += c

    results["avg_points"] = results["total_points"] / n_trials
    results["prob_A_wins"] = results["A_Wins"] / n_trials
    results["prob_B_wins"] = results["B_Wins"] / n_trials
    results["avg_state_visits"] = {
        s: results["state_visits"][s] / n_trials for s in transient_states
    }

    total_games = float(n_trials)
    results["length_dist"] = {
        pts: count / total_games
        for pts, count in sorted(results["length_counts"].items())
    }

    return results

# ---------------------------
# Streamlit app
# ---------------------------

def main():
    st.set_page_config(page_title="Volleyball Markov Chain Model", layout="wide")

    st.title("Volleyball to 5: Markov Chain Dual-Engine Validator")

    st.markdown(
        """
This app models a **race-to-5 volleyball game** with **win-by-2, no cap** as an
**absorbing Markov chain**, and validates it with Monte Carlo simulation.

- **Sprint Phase**: absolute scores (0–0, 1–0, …) before 3–3.
- **Duel Phase**: relative states -1, 0, +1 once the game is close.
        """
    )

    # Explicit required inputs section
    st.subheader("Required User Inputs")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**State Names**")
        st.markdown("- Transient: sprint states (score pairs) and duel states (-1, 0, +1).")
        st.markdown("- Absorbing: A_Wins, B_Wins.")

    with colB:
        st.markdown("**Transition Probabilities (p)**")
        st.markdown(
            "Single parameter p = P(Team A wins a point); "
            "all entries of the transition matrix P are built from p and (1-p)."
        )

    st.markdown(
        "**State Transition Matrix (P)** and **Simulation Parameters** "
        "are constructed and displayed after running the analysis."
    )
    st.markdown("---")

    # Sidebar inputs
    st.sidebar.header("Parameters")
    p = st.sidebar.slider(
        "Probability Team A wins a point (p)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    n_trials = st.sidebar.number_input(
        "Number of simulation trials",
        min_value=1,
        max_value=1_000_000,
        value=10_000,
        step=1,
        help="Number of games to simulate for Monte Carlo validation",
    )

    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=999_999,
        value=0,
        help="Set 0 for varying runs, nonzero for reproducible results.",
    )

    if st.sidebar.button("Run Analysis", type="primary"):
        if seed != 0:
            random.seed(seed)
            np.random.seed(seed)

        with st.spinner("Building Markov chain model..."):
            all_states, transient_states, absorbing_states, P, Q, R, state_to_idx = build_volleyball_model(p)

        with st.spinner("Computing analytical solution..."):
            F, t, B = analytical_solution(Q, R)

        with st.spinner(f"Running {n_trials:,} simulations..."):
            sim_results = run_simulations(p, n_trials, transient_states, absorbing_states)

        start_idx = state_to_idx["(0,0)"]

        st.success("Analysis complete.")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Summary",
                "State Space",
                "Matrices",
                "Analytical Results",
                "Simulation Results",
            ]
        )

        # ---------------- Summary tab ----------------
        with tab1:
            st.header("Summary: Theoretical vs Experimental")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Expected points (Theoretical)", f"{t[start_idx]:.2f}",
                )
                st.metric(
                    "Expected points (Experimental)", f"{sim_results['avg_points']:.2f}",
                    delta=f"{sim_results['avg_points'] - t[start_idx]:.2f}",
                )

            with col2:
                st.metric(
                    "P(Team A wins) - Theoretical", f"{B[start_idx, 0]:.4f}",
                )
                st.metric(
                    "P(Team A wins) - Experimental", f"{sim_results['prob_A_wins']:.4f}",
                    delta=f"{sim_results['prob_A_wins'] - B[start_idx, 0]:.4f}",
                )

            with col3:
                st.metric(
                    "P(Team B wins) - Theoretical", f"{B[start_idx, 1]:.4f}",
                )
                st.metric(
                    "P(Team B wins) - Experimental", f"{sim_results['prob_B_wins']:.4f}",
                    delta=f"{sim_results['prob_B_wins'] - B[start_idx, 1]:.4f}",
                )

            st.markdown("---")

            st.subheader("Convergence table")
            conv_df = pd.DataFrame(
                {
                    "Metric": ["Expected points", "P(A wins)", "P(B wins)"],
                    "Theoretical": [t[start_idx], B[start_idx, 0], B[start_idx, 1]],
                    "Experimental": [
                        sim_results["avg_points"],
                        sim_results["prob_A_wins"],
                        sim_results["prob_B_wins"],
                    ],
                }
            )
            conv_df["Absolute error"] = (
                conv_df["Experimental"] - conv_df["Theoretical"]
            ).abs()
            st.dataframe(
                conv_df.style.format(
                    {
                        "Theoretical": "{:.4f}",
                        "Experimental": "{:.4f}",
                        "Absolute error": "{:.4f}",
                    }
                )
            )

        # ---------------- State space tab ----------------
        with tab2:
            st.header("State Space Definition")

            col1, col2 = st.columns(2)
            sprint = [s for s in transient_states if s.startswith("(")]
            duel = [s for s in transient_states if not s.startswith("(")]

            with col1:
                st.subheader("Sprint phase (score pairs)")
                st.write(f"Total: {len(sprint)} states")
                st.code("\n".join(sprint))

            with col2:
                st.subheader("Duel phase (relative states)")
                st.write(f"Total: {len(duel)} states")
                st.code("\n".join(duel))

            st.subheader("Absorbing states")
            st.code("\n".join(absorbing_states))

            st.info(
                f"Total states: {len(all_states)} = {len(transient_states)} transient + {len(absorbing_states)} absorbing."
            )

        # ---------------- Matrices tab ----------------
        with tab3:
            st.header("Transition Matrices")

            st.subheader("Full transition matrix P")
            df_P = pd.DataFrame(P, index=all_states, columns=all_states)
            st.dataframe(df_P.style.format("{:.3f}"), height=400)

            st.subheader("Q (transient → transient)")
            df_Q = pd.DataFrame(Q, index=transient_states, columns=transient_states)
            st.dataframe(df_Q.style.format("{:.3f}"), height=400)

            st.subheader("R (transient → absorbing)")
            df_R = pd.DataFrame(R, index=transient_states, columns=absorbing_states)
            st.dataframe(df_R.style.format("{:.3f}"))

        # ---------------- Analytical tab ----------------
        with tab4:
            st.header("Analytical Results")

            st.subheader("Fundamental matrix F = (I - Q)^(-1)")
            df_F = pd.DataFrame(F, index=transient_states, columns=transient_states)
            st.dataframe(df_F.style.format("{:.4f}"), height=400)

            st.subheader("Expected steps to absorption (from each state)")
            df_t = pd.DataFrame(
                {"State": transient_states, "Expected points remaining": t}
            )
            st.dataframe(df_t.style.format({"Expected points remaining": "{:.4f}"}))

            st.subheader("Absorption probabilities B = F · R")
            df_B = pd.DataFrame(B, index=transient_states, columns=absorbing_states)
            st.dataframe(df_B.style.format("{:.4f}"))

        # ---------------- Simulation tab ----------------
        with tab5:
            st.header("Simulation Results (Monte Carlo)")

            st.subheader(f"Aggregate results from {n_trials:,} games")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Team A wins",
                    f"{sim_results['A_Wins']:,}",
                    f"{100 * sim_results['prob_A_wins']:.2f}%",
                )
            with c2:
                st.metric(
                    "Team B wins",
                    f"{sim_results['B_Wins']:,}",
                    f"{100 * sim_results['prob_B_wins']:.2f}%",
                )

            st.metric("Average game length", f"{sim_results['avg_points']:.2f} points")

            st.subheader("Game-length probability distribution")
            length_df = pd.DataFrame(
                {
                    "Total points": list(sim_results["length_dist"].keys()),
                    "Probability": list(sim_results["length_dist"].values()),
                }
            )
            st.dataframe(
                length_df.style.format({"Probability": "{:.4f}"}), height=300
            )

            if len(length_df) > 0:
                st.bar_chart(length_df.set_index("Total points"))

            st.markdown(
                "This distribution answers: *What is the probability the game "
                "ends in exactly N total points?*"
            )

            st.subheader("Average state visits per game")
            visits_df = pd.DataFrame(
                {
                    "State": transient_states,
                    "Avg visits (experimental)": [
                        sim_results["avg_state_visits"][s] for s in transient_states
                    ],
                    "Avg visits (theoretical)": [
                        F[start_idx, i] for i in range(len(transient_states))
                    ],
                }
            )
            visits_df["Absolute error"] = (
                visits_df["Avg visits (experimental)"]
                - visits_df["Avg visits (theoretical)"]
            ).abs()
            st.dataframe(
                visits_df.style.format(
                    {
                        "Avg visits (experimental)": "{:.4f}",
                        "Avg visits (theoretical)": "{:.4f}",
                        "Absolute error": "{:.4f}",
                    }
                ),
                height=400,
            )

if __name__ == "__main__":
    main()


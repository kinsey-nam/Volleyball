
import streamlit as st
import numpy as np
import pandas as pd
import random

def build_volleyball_model(p):
    """
    Build the Markov Chain transition matrix for volleyball to 5 (win by 2).

    States:
    - Before 3-3: Absolute scores (a,b) where a,b in {0,1,2,3,4}
    - At 3-3 and beyond: Relative states (-1, 0, +1) representing differential
    - Absorbing: "A_Wins", "B_Wins"

    Returns state_names, P (full transition matrix), Q, R
    """

    # Define all states
    # Phase 1: "Sprint" - absolute scores before entering close game
    sprint_states = []
    for a in range(5):
        for b in range(5):
            # Exclude states that are already won or trigger close-game
            if a >= 5 or b >= 5:
                continue
            if (a == 3 and b == 3):  # This enters close game
                continue
            if (a >= 5 or b >= 5) and abs(a - b) >= 2:
                continue
            sprint_states.append(f"({a},{b})")

    # Phase 2: "Duel" - close game states (relative differential)
    duel_states = ["-1", "0", "+1"]

    # Absorbing states
    absorbing_states = ["A_Wins", "B_Wins"]

    # All transient states
    transient_states = sprint_states + duel_states
    all_states = transient_states + absorbing_states

    n_transient = len(transient_states)
    n_total = len(all_states)

    # Create state index mapping
    state_to_idx = {state: idx for idx, state in enumerate(all_states)}

    # Initialize transition matrix
    P = np.zeros((n_total, n_total))

    # Fill in transitions
    for i, state in enumerate(all_states):
        if state in absorbing_states:
            # Absorbing states stay in themselves
            P[i, i] = 1.0
            continue

        # Transient state - determine next states
        if state in sprint_states:
            # Parse score
            a, b = map(int, state.strip("()").split(","))

            # Team A wins point (probability p)
            next_a = a + 1
            next_b = b

            # Check if this leads to a win
            if next_a >= 5 and next_a - next_b >= 2:
                P[i, state_to_idx["A_Wins"]] += p
            # Check if this enters close game (3-3 becomes 0)
            elif next_a == 3 and next_b == 3:
                P[i, state_to_idx["0"]] += p
            elif next_a == 4 and next_b == 3:
                P[i, state_to_idx["+1"]] += p
            elif next_a == 3 and next_b == 4:
                P[i, state_to_idx["-1"]] += p
            else:
                # Stay in sprint
                next_state = f"({next_a},{next_b})"
                if next_state in state_to_idx:
                    P[i, state_to_idx[next_state]] += p

            # Team B wins point (probability 1-p)
            next_a = a
            next_b = b + 1

            if next_b >= 5 and next_b - next_a >= 2:
                P[i, state_to_idx["B_Wins"]] += (1 - p)
            elif next_a == 3 and next_b == 3:
                P[i, state_to_idx["0"]] += (1 - p)
            elif next_a == 4 and next_b == 3:
                P[i, state_to_idx["+1"]] += (1 - p)
            elif next_a == 3 and next_b == 4:
                P[i, state_to_idx["-1"]] += (1 - p)
            else:
                next_state = f"({next_a},{next_b})"
                if next_state in state_to_idx:
                    P[i, state_to_idx[next_state]] += (1 - p)

        elif state in duel_states:
            # In duel phase - only differential matters
            diff = int(state) if state != "0" else 0

            # Team A wins point
            new_diff = diff + 1
            if new_diff >= 2:
                P[i, state_to_idx["A_Wins"]] += p
            elif new_diff == 1:
                P[i, state_to_idx["+1"]] += p
            elif new_diff == 0:
                P[i, state_to_idx["0"]] += p
            elif new_diff == -1:
                P[i, state_to_idx["-1"]] += p

            # Team B wins point
            new_diff = diff - 1
            if new_diff <= -2:
                P[i, state_to_idx["B_Wins"]] += (1 - p)
            elif new_diff == 1:
                P[i, state_to_idx["+1"]] += (1 - p)
            elif new_diff == 0:
                P[i, state_to_idx["0"]] += (1 - p)
            elif new_diff == -1:
                P[i, state_to_idx["-1"]] += (1 - p)

    # Partition into Q and R
    Q = P[:n_transient, :n_transient]
    R = P[:n_transient, n_transient:]

    return all_states, transient_states, absorbing_states, P, Q, R, state_to_idx


def analytical_solution(Q, R):
    """
    Compute analytical solutions using matrix algebra.
    F = (I - Q)^-1 (Fundamental Matrix)
    B = F * R (Absorption probabilities)
    """
    n = Q.shape[0]
    I = np.eye(n)

    # Fundamental matrix
    F = np.linalg.inv(I - Q)

    # Expected number of steps from each starting state
    t = F.sum(axis=1)

    # Absorption probabilities
    B = F @ R

    return F, t, B


def simulate_game(p, state_to_idx, transient_states, absorbing_states):
    """
    Simulate a single volleyball game using Monte Carlo method.
    Returns: final absorbing state, total points played, states visited
    """
    current_state = "(0,0)"
    points = 0
    states_visited = {state: 0 for state in transient_states}

    max_points = 1000  # Safety limit

    while current_state not in absorbing_states and points < max_points:
        states_visited[current_state] += 1

        # Determine next state
        if current_state.startswith("("):
            # Sprint phase
            a, b = map(int, current_state.strip("()").split(","))

            if random.random() < p:
                # A wins point
                a += 1
            else:
                # B wins point
                b += 1

            points += 1

            # Check for win
            if a >= 5 and a - b >= 2:
                current_state = "A_Wins"
            elif b >= 5 and b - a >= 2:
                current_state = "B_Wins"
            # Check for entry into duel
            elif a == 3 and b == 3:
                current_state = "0"
            elif a == 4 and b == 3:
                current_state = "+1"
            elif a == 3 and b == 4:
                current_state = "-1"
            else:
                current_state = f"({a},{b})"

        else:
            # Duel phase
            diff = int(current_state) if current_state != "0" else 0

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


def run_simulations(p, n_trials, state_to_idx, transient_states, absorbing_states):
    """Run Monte Carlo simulations."""
    results = {
        "A_Wins": 0,
        "B_Wins": 0,
        "total_points": 0,
        "state_visits": {state: 0 for state in transient_states}
    }

    for _ in range(n_trials):
        final_state, points, visits = simulate_game(p, state_to_idx, transient_states, absorbing_states)

        if final_state == "A_Wins":
            results["A_Wins"] += 1
        elif final_state == "B_Wins":
            results["B_Wins"] += 1

        results["total_points"] += points

        for state, count in visits.items():
            results["state_visits"][state] += count

    # Calculate averages
    results["avg_points"] = results["total_points"] / n_trials
    results["prob_A_wins"] = results["A_Wins"] / n_trials
    results["prob_B_wins"] = results["B_Wins"] / n_trials
    results["avg_state_visits"] = {state: count / n_trials for state, count in results["state_visits"].items()}

    return results


def main():
    st.set_page_config(page_title="Volleyball Markov Chain Model", layout="wide")

    st.title("🏐 Volleyball to 5: Markov Chain Dual-Engine Validator")

    st.markdown("""
    This tool validates a **race-to-5 volleyball game** with **win-by-2** using both:
    - **Analytical Engine**: Matrix algebra (absorbing Markov chains)
    - **Simulation Engine**: Monte Carlo methods

    The game has two phases:
    - **Sprint Phase**: Absolute scores before 3-3 (binary tree structure)
    - **Duel Phase**: Relative differential states after 3-3 (-1, 0, +1)
    """)

    # User inputs
    st.sidebar.header("⚙️ Parameters")
    p = st.sidebar.slider(
        "Probability Team A wins a point (p)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Probability that Team A wins any given point"
    )

    n_trials = st.sidebar.number_input(
        "Number of simulation trials",
        min_value=100,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Number of games to simulate for Monte Carlo validation"
    )

    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        help="Set seed for reproducibility"
    )

    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        random.seed(seed)
        np.random.seed(seed)

        with st.spinner("Building Markov Chain model..."):
            all_states, transient_states, absorbing_states, P, Q, R, state_to_idx = build_volleyball_model(p)

        # Analytical solution
        with st.spinner("Computing analytical solution..."):
            F, t, B = analytical_solution(Q, R)

        # Simulation
        with st.spinner(f"Running {n_trials:,} simulations..."):
            sim_results = run_simulations(p, n_trials, state_to_idx, transient_states, absorbing_states)

        # Display results
        st.success("Analysis complete!")

        # Tabs for organized output
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Summary",
            "🎯 State Space",
            "🔢 Matrices",
            "📈 Analytical Results",
            "🎲 Simulation Results"
        ])

        with tab1:
            st.header("Summary: Theoretical vs Experimental")

            col1, col2, col3 = st.columns(3)

            # Starting state is (0,0), which is index 0
            start_idx = state_to_idx["(0,0)"]

            with col1:
                st.metric(
                    "Expected Points (Theoretical)",
                    f"{t[start_idx]:.2f}",
                    help="Average game length from (0,0)"
                )
                st.metric(
                    "Expected Points (Experimental)",
                    f"{sim_results['avg_points']:.2f}",
                    delta=f"{sim_results['avg_points'] - t[start_idx]:.2f}"
                )

            with col2:
                st.metric(
                    "P(Team A Wins) - Theoretical",
                    f"{B[start_idx, 0]:.4f}",
                    help="Probability Team A wins from (0,0)"
                )
                st.metric(
                    "P(Team A Wins) - Experimental",
                    f"{sim_results['prob_A_wins']:.4f}",
                    delta=f"{sim_results['prob_A_wins'] - B[start_idx, 0]:.4f}"
                )

            with col3:
                st.metric(
                    "P(Team B Wins) - Theoretical",
                    f"{B[start_idx, 1]:.4f}",
                    help="Probability Team B wins from (0,0)"
                )
                st.metric(
                    "P(Team B Wins) - Experimental",
                    f"{sim_results['prob_B_wins']:.4f}",
                    delta=f"{sim_results['prob_B_wins'] - B[start_idx, 1]:.4f}"
                )

            st.markdown("---")
            st.subheader("Convergence Analysis")

            convergence_data = {
                "Metric": ["Expected Points", "P(A Wins)", "P(B Wins)"],
                "Theoretical": [t[start_idx], B[start_idx, 0], B[start_idx, 1]],
                "Experimental": [
                    sim_results['avg_points'],
                    sim_results['prob_A_wins'],
                    sim_results['prob_B_wins']
                ],
                "Absolute Error": [
                    abs(sim_results['avg_points'] - t[start_idx]),
                    abs(sim_results['prob_A_wins'] - B[start_idx, 0]),
                    abs(sim_results['prob_B_wins'] - B[start_idx, 1])
                ],
                "Relative Error (%)": [
                    100 * abs(sim_results['avg_points'] - t[start_idx]) / t[start_idx],
                    100 * abs(sim_results['prob_A_wins'] - B[start_idx, 0]) / B[start_idx, 0] if B[start_idx, 0] > 0 else 0,
                    100 * abs(sim_results['prob_B_wins'] - B[start_idx, 1]) / B[start_idx, 1] if B[start_idx, 1] > 0 else 0
                ]
            }

            st.dataframe(pd.DataFrame(convergence_data), use_container_width=True)

        with tab2:
            st.header("State Space Definition")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sprint Phase States")
                sprint = [s for s in transient_states if s.startswith("(")]
                st.write(f"Total: {len(sprint)} states")
                st.code("\n".join(sprint))

            with col2:
                st.subheader("Duel Phase States")
                duel = [s for s in transient_states if not s.startswith("(")]
                st.write(f"Total: {len(duel)} states")
                st.code("\n".join(duel))

            st.subheader("Absorbing States")
            st.code("\n".join(absorbing_states))

            st.info(f"**Total States**: {len(all_states)} ({len(transient_states)} transient + {len(absorbing_states)} absorbing)")

        with tab3:
            st.header("Transition Matrices")

            st.subheader("Full Transition Matrix (P)")
            df_P = pd.DataFrame(P, index=all_states, columns=all_states)
            st.dataframe(df_P.style.format("{:.3f}"), height=400)

            st.subheader("Q Matrix (Transient → Transient)")
            df_Q = pd.DataFrame(Q, index=transient_states, columns=transient_states)
            st.dataframe(df_Q.style.format("{:.3f}"), height=400)

            st.subheader("R Matrix (Transient → Absorbing)")
            df_R = pd.DataFrame(R, index=transient_states, columns=absorbing_states)
            st.dataframe(df_R.style.format("{:.3f}"))

        with tab4:
            st.header("Analytical Results (Matrix-Based)")

            st.subheader("Fundamental Matrix F = (I - Q)⁻¹")
            df_F = pd.DataFrame(F, index=transient_states, columns=transient_states)
            st.dataframe(df_F.style.format("{:.4f}"), height=400)

            st.markdown("The Fundamental Matrix shows the expected number of times the chain visits state j given it starts in state i.")

            st.subheader("Expected Steps to Absorption (t)")
            df_t = pd.DataFrame({
                "State": transient_states,
                "Expected Points Remaining": t
            })
            st.dataframe(df_t.style.format({"Expected Points Remaining": "{:.4f}"}))

            st.subheader("Absorption Probability Matrix B = F × R")
            df_B = pd.DataFrame(B, index=transient_states, columns=absorbing_states)
            st.dataframe(df_B.style.format("{:.4f}"))

            st.markdown("Each row shows the probability of reaching each absorbing state from that starting state.")

        with tab5:
            st.header("Simulation Results (Monte Carlo)")

            st.subheader(f"Results from {n_trials:,} trials")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Team A Wins", f"{sim_results['A_Wins']:,}", f"{100*sim_results['prob_A_wins']:.2f}%")
            with col2:
                st.metric("Team B Wins", f"{sim_results['B_Wins']:,}", f"{100*sim_results['prob_B_wins']:.2f}%")

            st.metric("Average Game Length", f"{sim_results['avg_points']:.2f} points")

            st.subheader("Average State Visits per Game")
            visits_df = pd.DataFrame({
                "State": list(sim_results['avg_state_visits'].keys()),
                "Avg Visits (Experimental)": list(sim_results['avg_state_visits'].values()),
                "Avg Visits (Theoretical)": [F[start_idx, state_to_idx[state]] for state in transient_states]
            })
            visits_df["Absolute Error"] = abs(visits_df["Avg Visits (Experimental)"] - visits_df["Avg Visits (Theoretical)"])

            st.dataframe(
                visits_df.style.format({
                    "Avg Visits (Experimental)": "{:.4f}",
                    "Avg Visits (Theoretical)": "{:.4f}",
                    "Absolute Error": "{:.4f}"
                }),
                height=400
            )

if __name__ == "__main__":
    main()

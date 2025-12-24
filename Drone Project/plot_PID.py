import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_flight_data():
    try:
        # Read the CSV file by tracker
        df = pd.read_csv("flight_metrics.csv")
    except FileNotFoundError:
        print("Error: 'flight_metrics.csv' not found. Run your tracker first!")
        return

    # Normalize time to start at 0 seconds
    start_time = df['timestamp'].iloc[0]
    df['relative_time'] = df['timestamp'] - start_time

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Tracking Error (RMSE) plot
    ax1.plot(df['relative_time'], df['error_x'], label='Lateral Error', color='blue')
    ax1.plot(df['relative_time'], df['error_y'], label='Vertical Error', color='orange', alpha=0.7)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title("Tracking Accuracy (Closer to 0 is better)")
    ax1.set_ylabel("Normalized Error (-1 to 1)")
    ax1.legend()
    ax1.grid(True)

    # Distance plot
    ax2.plot(df['relative_time'], df['dist'], label='Target Distance', color='green')
    ax2.axhline(12.0, color='red', linestyle='--', label='Setpoint (12m)')
    ax2.set_title("Distance Control")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Distance (meters)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_flight_data()
import matplotlib
import matplotlib.pyplot as plt
import polars as pl

matplotlib.use("Agg")


def main() -> None:
    df_step = pl.read_parquet("simulation.parquet")
    df_adv  = pl.read_parquet("simulation_advance.parquet")

    plt.plot(df_step["time_s"], df_step["speed_kmh"],
             label='step_trains (dt = 0.1 s)', linewidth=1)
    plt.plot(df_adv["time_s"], df_adv["speed_kmh"],
             label='advance_train (dt = 100 s)', marker='o', linestyle='--', linewidth=1.5)

    plt.xlabel('time (s)')
    plt.ylabel('speed (km/h)')
    plt.legend()
    plt.title('step_trains vs advance_train')
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)


if __name__ == "__main__":
    main()

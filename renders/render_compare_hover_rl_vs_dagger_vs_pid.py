"""Compare hover performance: PPO(RL) vs DAgger vs PID.

This entrypoint keeps the PID baseline enabled while reusing the main
RL-vs-DAgger renderer implementation.
"""
import sys

from render_compare_hover_rl_vs_dagger import main


if __name__ == '__main__':
    if '--with-pid' not in sys.argv:
        sys.argv.append('--with-pid')
    main()

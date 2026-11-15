"""Train the team's [thrust,roll,pitch,yaw] -> [az,wdotz] RNN."""

from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.training.train import main


if __name__ == '__main__':
    main(['--mode', 'z_psi', *sys.argv[1:]])

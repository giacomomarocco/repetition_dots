"""Export the small Unicode browser viewer from completed analysis artifacts."""
import argparse
from pathlib import Path
from filler.dsv4.recurrence_viewer import write_viewer

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    print(write_viewer(parser.parse_args().output))

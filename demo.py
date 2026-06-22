#!/usr/bin/env python3

import argparse

from HandTrackerEdge import HandTrackerEdge
from HandTrackerRenderer import HandTrackerRenderer

parser = argparse.ArgumentParser(description="pib hand imitation via OAK camera (Edge mode)")
parser.add_argument('-o', '--output', help="Path to output video file")
parser.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0,
                    help="Print debug info (optional trace level)")
parser.add_argument('--debug-thumbs', action='store_true',
                    help='Print thumb motor values to the console while imitating')
parser.add_argument('--debug-thumbs-every', type=int, default=10, metavar='N',
                    help='With --debug-thumbs, print every N frames (default: 10)')
args = parser.parse_args()

tracker = HandTrackerEdge(
    stats=True,
    trace=args.trace,
    debug_thumbs=args.debug_thumbs,
    debug_thumbs_interval=args.debug_thumbs_every,
)
renderer = HandTrackerRenderer(tracker=tracker, output=args.output)

while True:
    frame, hands, bag = tracker.next_frame()
    if frame is None:
        break
    frame = renderer.draw(frame, hands, bag)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break

renderer.exit()
tracker.exit()

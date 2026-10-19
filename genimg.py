"""生图 agent 入口。

用法示例
--------
    python genimg.py "a red panda astronaut, cinematic" --size 512x512
    python genimg.py "水墨山水，留白 [精度=fp32] [画质=max]"
    python genimg.py --image in.png --strength 0.6 "油画风格"
    python genimg.py --list
"""

import sys

from genimg.cli import main

if __name__ == "__main__":
    sys.exit(main())

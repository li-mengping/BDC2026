"""仓库侧车入口；实现由产品模块统一拥有。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from code.utils.data_manifest import main


if __name__ == '__main__':
    main()

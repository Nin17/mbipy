# DATA

>[!TIP]
>To download the data for the examples, you need to use [git-lfs](https://git-lfs.com).

## BeLens

Beryllium lenses (with a radius of curvature of 50μm, 500μm, 1500μm and 4000μm respectively) imaged at BM05 of the ESRF.

* "50"

    |    Keys     |     Shape      | Dtype  |
    | :---------: | :------------: | :----: |
    | "reference" | (900, 900, 10) | uint16 |
    |  "sample"   | (900, 900, 10) | uint16 |

* "500"

    |    Keys     |      Shape       | Dtype  |
    | :---------: | :--------------: | :----: |
    | "reference" | (2048, 2048, 10) | uint16 |
    |  "sample"   | (2048, 2048, 10) | uint16 |

* "1500"

    |    Keys     |      Shape       | Dtype  |
    | :---------: | :--------------: | :----: |
    | "reference" | (2048, 2048, 10) | uint16 |
    |  "sample"   | (2048, 2048, 10) | uint16 |

* "4000"

    |    Keys     |      Shape       | Dtype  |
    | :---------: | :--------------: | :----: |
    | "reference" | (2048, 2048, 10) | uint16 |
    |  "sample"   | (2048, 2048, 10) | uint16 |

## Phantom

Tomographic phantom from Scientific Solutions imaged at ID17 of the ESRF.

Propagation distance 3.545m, pixel size 13.4μm, energies in keV.

|    Keys     |       Shape        | Dtype  |
| :---------: | :----------------: | :----: |
| "energies"  |      (10, 5)       | uint16 |
| "reference" | (10, 611, 2560, 5) | uint16 |
|  "sample"   | (10, 611, 2560, 5) | uint16 |

## Sandpaper

Sandpaper imaged at Xenocs at different distances: 120cm, 180cm and 320cm.

* "3M"

    | Keys  |      Shape      |  Dtype  |
    | :---: | :-------------: | :-----: |
    | "320" | (1700, 928, 10) | float32 |

* "dext"

    | Keys  |      Shape      |  Dtype  |
    | :---: | :-------------: | :-----: |
    | "120" | (1700, 928, 10) | float32 |
    | "180" | (1700, 928, 10) | float32 |
    | "320" | (1700, 928, 10) | float32 |

## WattleFlower

Wattle flower imaged at hutch 3B of the Imaging and Medical Beamline (IMBL) at the
Australian Synchrotron [[1]](#1).

|    Keys     |      Shape       |  Dtype  |
| :---------: | :--------------: | :-----: |
|   "dark"    | (2157, 2560, 15) | float32 |
| "reference" | (2157, 2560, 15) | float32 |
|  "sample"   | (2157, 2560, 15) | uint16  |
|   "white"   |   (2157, 2560)   | float32 |

## WireID17

|    Keys     |      Shape       | Dtype  |
| :---------: | :--------------: | :----: |
| "reference" | (2160, 2560, 10) | uint16 |
|  "sample"   | (2160, 2560, 10) | uint16 |

## WireXenocs

* "400"

    |    Keys     |      Shape      |  Dtype  |
    | :---------: | :-------------: | :-----: |
    | "reference" | (1028, 512, 10) | float32 |
    |  "sample"   | (1028, 512, 10) | float32 |

* "800"

    |    Keys     |      Shape      |  Dtype  |
    | :---------: | :-------------: | :-----: |
    | "reference" | (1030, 514, 10) | float32 |
    |  "sample"   | (1030, 514, 10) | float32 |

* "1200"

    |    Keys     |      Shape      |  Dtype  |
    | :---------: | :-------------: | :-----: |
    | "reference" | (1030, 514, 10) | float32 |
    |  "sample"   | (1030, 514, 10) | float32 |

* "2200"

    |    Keys     |      Shape      |  Dtype  |
    | :---------: | :-------------: | :-----: |
    | "reference" | (1030, 514, 10) | float32 |
    |  "sample"   | (1030, 514, 10) | float32 |

* "3200"

    |    Keys     |      Shape      |  Dtype  |
    | :---------: | :-------------: | :-----: |
    | "reference" | (1030, 514, 10) | float32 |
    |  "sample"   | (1030, 514, 10) | float32 |

## References

<a id="1">[1]</a>
Alloo, S.J., Morgan, K.S., Paganin, D.M. et al. M﻿ultimodal intrinsic speckle-tracking (MIST) to extract images of rapidly-varying diffuse X-ray dark-field. Sci Rep 13, 5424 (2023). <https://doi.org/10.1038/s41598-023-31574-z>

#!/bin/bash
clss=('carpet' 'leather' 'grid' 'tile' 'wood' 'bottle' 'cable' 'capsule' 'hazelnut' 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'zipper')
for i in {0..14}
    do
        python test.py --defect_cls ${clss[${i}]}
    done
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamfdb_tracker import SiameseTracker
from pysot.tracker.siamfdb_tracker import SiamFDBTracker

TRACKS = {
          'SiamFDBTracker': SiamFDBTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)

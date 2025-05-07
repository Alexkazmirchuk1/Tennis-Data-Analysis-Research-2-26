import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager
import pathlib

# Computer-specific...
#import matplotlib
#matplotlib.use('Qt5Agg')
#

font_path = "font/Atkinson Hyperlegible Next/AtkinsonHyperlegibleNext-Medium.otf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

fontpath = pathlib.Path(matplotlib.get_data_path(), "Atkinson Hyperlegible Mono/AtkinsonHyperlegibleMono-Medium.otf")


plt.rcParams.update(
    {'font.size': 13,
    #'font.family': 'sans-serif',
    #'font.sans-serif': 'Geneva'
    'font.family': 'monospace',
    'font.monospace': prop.get_name()
    }
)


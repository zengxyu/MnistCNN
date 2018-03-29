DATA_PATH = "data"
IMAGE_tmp = "image"
DATA_DIR = "data/data_train"

IMAGE_WIDTH = 28#图像的宽
IMAGE_HEIGHT = 28#图像的高
LABEL_BYTES = 1
NUM_CLASSES = 10    #手写数字识别共有十个类别

NUM_EXAMPLES_FOR_TRAIN = 60000
BATCH_SIZE = 50  # 每一批次处理的样本的数量为128

LOG_FREQUENCY = 10#打印频率
MAX_STEPS = 100000#最大迭代次数